"""容器训练入口与 JSONL 预测入口。"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import Model, RoundEngine, checkpoint_digest, eval_checkpoint, get_llm_cfg, read_rows, write_rows
from baselines.data import check_dataset_file, flatten, json_value
from baselines.predict import predict

TASKS = ("A", "B", "C")


def checkpoint_at(root, task):
    if root.is_file():
        return root
    candidates = (root / task / "checkpoint.npz",
                  root / "model" / task / "checkpoint.npz",
                  root / "output" / "model" / task / "checkpoint.npz",
                  root / f"{task}-baseline.npz",
                  root / "model" / "checkpoint.npz", root / "checkpoint.npz")
    return next((path for path in candidates if path.is_file()), None)


def parent_checkpoint(args, task):
    checkpoint = args.checkpoint or checkpoint_at(args.previous, task)
    if checkpoint is None:
        # 赛方也可把训练集和起始检查点放在同一任务目录。
        checkpoint = checkpoint_at(args.data if args.task != "all" else args.data / task, task)
    if checkpoint is None:
        raise FileNotFoundError(f"任务 {task} 缺少平台起始检查点，请挂载 {args.previous} 或指定 --checkpoint")
    model = Model.load(checkpoint)
    if model.meta["profile"] != task:
        raise ValueError(f"任务 {task} 起始检查点的任务为 {model.meta['profile']}")
    return checkpoint


def task_data(args, task):
    directory = args.data / task
    if args.task != "all" and not directory.is_dir():
        directory = args.data
    return directory / "public" if (directory / "public").is_dir() else directory


def equipment(task, row):
    meta = row.get("meta", {})
    if task == "B":
        return meta.get("bearing")
    if task == "C" and meta.get("source_subset") is not None and meta.get("unit") is not None:
        return (meta["source_subset"], meta["unit"])
    return None


def public_batches(task, rows, rounds):
    if len(rows) < 3 * rounds:
        raise ValueError(f"任务 {task} 的公开训练样本不足以划分 {rounds} 轮")
    groups = defaultdict(list)
    for row in rows:
        group = row.get("meta", {}).get("source_batch") if task == "A" else equipment(task, row)
        groups[group].append(row)
    batches = [[] for _ in range(rounds)]
    if task == "A" and len(groups) == 1:
        # 一份公开训练文件只有一个采集阶段时，按类释放样本，避免小批次只有部分气体。
        class_counts = defaultdict(int)
        for row in rows:
            label = row["answer"]
            batches[class_counts[label] % rounds].append(row)
            class_counts[label] += 1
        return batches
    if task == "C" and None not in groups and len(groups) >= rounds:
        # 整台发动机进入同一批次，每个工况子集轮流分配。
        subset_counts = defaultdict(int)
        for subset, unit in sorted(groups):
            batches[subset_counts[subset] % rounds].extend(groups[(subset, unit)])
            subset_counts[subset] += 1
        return batches

    stages = list(groups.values())
    while len(stages) < rounds:
        largest = max(range(len(stages)), key=lambda i: len(stages[i]))
        stage = stages[largest]
        if len(stage) < 6:
            raise ValueError(f"任务 {task} 的公开训练样本不足以划分 {rounds} 轮")
        middle = len(stage) // 2
        stages[largest:largest + 1] = [stage[:middle], stage[middle:]]
    for i, stage in enumerate(stages):
        batches[i * rounds // len(stages)].extend(stage)
    return batches


def training_batches(args, task, directory, work):
    for folder in (directory, directory / "batches", directory / "rounds"):
        numbered = [(int(match[1]), path) for path in folder.glob("batch*.jsonl")
                    if (match := re.fullmatch(r"batch(\d+)\.jsonl", path.name))]
        if numbered:
            numbered.sort()
            numbers = [number for number, _ in numbered]
            if [number for number in numbers if number <= args.rounds] != list(range(1, args.rounds + 1)):
                raise ValueError(f"任务 {task} 需提供连续的 batch01 至 batch{args.rounds:02d}")
            paths = [path for _, path in numbered[:args.rounds]]
            for path in paths:
                check_dataset_file(path)
            break
    else:
        train = directory / "train.jsonl"
        check_dataset_file(train)
        batches = public_batches(task, read_rows(train), args.rounds)
        paths = []
        for index, rows in enumerate(batches, 1):
            if len(rows) < 3:
                raise ValueError(f"任务 {task} 第 {index} 轮的训练样本不足")
            path = work / "stream" / f"batch{index:02d}.jsonl"
            write_rows(path, rows)
            paths.append(path)

    validation = directory / "validation.jsonl"
    if validation.is_file():
        check_dataset_file(validation)
        held_out = read_rows(validation)
        held_out_ids = {row["id"] for row in held_out}
        held_out_devices = {device for row in held_out if (device := equipment(task, row)) is not None}
        for path in paths:
            if any(row["id"] in held_out_ids or equipment(task, row) in held_out_devices
                   for row in read_rows(path)):
                raise ValueError(f"任务 {task} 的训练批次 {path.name} 与公开验证集重叠")
    return paths


def deliver(args, task, parent, summary):
    source = Path(summary["adopted_final_ckpt"])
    digest = checkpoint_digest(source)
    if digest != summary["adopted_final_digest"]:
        raise ValueError(f"任务 {task} 最终检查点与训练记录的校验值不符")
    parent_digest = checkpoint_digest(parent)
    held = digest == parent_digest
    trained_from = parent_digest
    if not held:
        report = json_value((source.parent / "training.json").read_text(encoding="utf-8"))
        trained_from = report["parent_checkpoint_sha256"]
    rows = summary["rounds"]
    manifest = {
        "task": task, "status": "completed", "delivered": True, "failure": None,
        "kind": "classification" if task == "A" else "regression",
        "primary": "macro_f1" if task == "A" else "rmse",
        "origin": "baseline_kept" if held else "evolved",
        "source": str(source), "bytes": source.stat().st_size,
        "sha256": digest, "initial_sha256": parent_digest, "parent_sha256": trained_from,
        "initial_checkpoint": str(parent), "rounds_completed": summary["decision_count"],
        "policy": args.policy, "stream_mode": "train",
        "metric_series": [row["adopted_after_primary"] for row in rows],
        "promotions": sum(row["decision"] == "DEPLOY_NEW" for row in rows),
        "experiment_count": summary["experiment_count"], "llm_rounds": summary["llm_rounds"],
        "llm_tool_rounds": summary["llm_tool_rounds"], "stage": summary["adopted_final_stage"],
        "target_unit": {"A": "class", "B": "minutes", "C": "cycles"}[task],
        "summary": str(args.out / task / "summary.json"),
    }
    for root in dict.fromkeys((args.model_out.resolve(), (args.out / "model").resolve())):
        destination = root / task
        destination.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination) as staging:
            temp = Path(staging)
            shutil.copyfile(source, temp / "checkpoint.npz")
            document = {**manifest, "checkpoint": str(destination / "checkpoint.npz")}
            (temp / "manifest.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            for name in ("checkpoint.npz", "manifest.json"):
                (temp / name).replace(destination / name)


def load_models(args, tasks):
    explicit = args.checkpoint
    if explicit:
        model = Model.load(explicit)
        if model.meta["profile"] not in tasks:
            raise ValueError("检查点与 --task 不一致")
        return {model.meta["profile"]: model}
    models = {}
    for task in tasks:
        checkpoint = checkpoint_at(args.model_out, task) or checkpoint_at(args.out / "model", task)
        if checkpoint is not None:
            model = Model.load(checkpoint)
            if model.meta["profile"] != task:
                raise ValueError(f"任务 {task} 的预测检查点与任务不一致")
            models[task] = model
    if not models:
        raise FileNotFoundError("找不到已交付的预测检查点，请挂载 /model 或指定 --checkpoint")
    return models


def serve(models):
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json_value(line)
        if not isinstance(request, dict) or set(request) - {"task", "inputs"} or "inputs" not in request:
            raise ValueError("预测请求只接受 inputs 和可选的 task")
        selector = request.get("task")
        inputs = request["inputs"]
        paths, _ = flatten(inputs)
        matches = [model for task, model in models.items()
                   if selector in (None, task, model.meta["task"]) and paths == model.meta["paths"]]
        if len(matches) != 1:
            raise ValueError("无法从任务和特征结构唯一确定预测检查点")
        print(json.dumps(predict(matches[0], inputs), ensure_ascii=False, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("loop", "serve", "selftest"), default=os.getenv("HARNESS_MODE", "loop"))
    parser.add_argument("--task", choices=("all", *TASKS), default="all")
    parser.add_argument("--data", type=Path, default=os.getenv("HARNESS_DATA", "/data"))
    parser.add_argument("--previous", type=Path, default=os.getenv("HARNESS_PREVIOUS", "/previous"))
    parser.add_argument("--model-out", type=Path, default=os.getenv("HARNESS_MODEL_OUT", "/model"))
    parser.add_argument("--out", type=Path, default="/output")
    parser.add_argument("--checkpoint", "--baseline", type=Path)
    parser.add_argument("--windows", "--rounds", dest="rounds", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--stream", choices=("auto", "train", "official"), default="auto")
    parser.add_argument("--policy", choices=("auto", "rule", "llm"), default="auto")
    parser.add_argument("--swan", choices=("auto", "off"), default="auto")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--replay-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-experiments", type=int, default=3)
    args = parser.parse_args()
    if args.mode not in ("loop", "serve", "selftest"):
        parser.error("HARNESS_MODE 需为 loop、serve 或 selftest")
    tasks = TASKS if args.task == "all" else (args.task,)

    if args.mode == "serve":
        serve(load_models(args, tasks))
        return
    if args.checkpoint and len(tasks) != 1:
        parser.error("训练或自检使用 --checkpoint 时需指定一个 --task")
    if args.swan == "off" or not (os.getenv("SWANLAB_MODE") or os.getenv("SWANLAB_API_KEY")):
        os.environ["SWANLAB_MODE"] = "disabled"
    llm = {} if args.policy == "rule" else None
    if args.policy == "llm" and not all(get_llm_cfg().values()):
        parser.error("--policy llm 需要 PyTRiO 地址、模型和密钥")

    parents = {task: parent_checkpoint(args, task) for task in tasks}
    for task, parent in parents.items():
        directory = task_data(args, task)
        if args.mode == "selftest":
            with TemporaryDirectory() as work:
                training_batches(args, task, directory, Path(work))
            print(f"{task} 输入自检通过 {checkpoint_digest(parent)}")
            continue
        if args.stream == "official":
            model = Model.load(parent)
            metrics = eval_checkpoint(parent, directory / "validation.jsonl", model.meta["task"], model.meta["classes"])
            print(json.dumps({"task": task, "mode": "validation_only", "metrics": metrics}, allow_nan=False))
            continue

        work = args.out / task
        work.mkdir(parents=True, exist_ok=False)
        paths = training_batches(args, task, directory, work)
        engine = RoundEngine(task, ROOT / "configs" / f"{task}.json", parent, paths, work,
                             rounds=args.rounds, val_ratio=args.val_ratio, replay_ratio=args.replay_ratio,
                             seed=args.seed, llm=llm, min_experiments=args.min_experiments)
        summary = engine.run()
        if summary["decision_count"] != args.rounds or any(not row["n_experiments"] for row in summary["rounds"]):
            raise RuntimeError(f"任务 {task} 未完成要求的训练和决策轮次，详见 {work}")
        if args.policy == "llm" and summary["llm_rounds"] < min(3, args.rounds):
            raise RuntimeError(f"任务 {task} 的 PyTRiO 规划轮数不足，详见 {work}")
        deliver(args, task, parent, summary)
        print(f"{task} 完成 {args.rounds} 轮，已交付 {summary['adopted_final_stage']}")


if __name__ == "__main__":
    main()
