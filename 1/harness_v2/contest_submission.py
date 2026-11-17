"""赛方 Harness SDK 1.0.2 的训练和预测入口。"""
from functools import lru_cache
import inspect
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

# 在导入 NumPy 之前设置官方基线的 CPU 线程数。
os.environ.setdefault("BASELINE_CPU_THREADS", "4")

from baselines.data import read_dataset
from baselines.model import Model, checkpoint_digest
from baselines.predict import predict as predict_checkpoint
from baselines.train import metrics, train_run
from pytrio_tools import parse_strategy


TASKS = ("classification", "regression")
ROOT = Path(__file__).resolve().parent
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
FALLBACK = {"A": (3, 0.001, 128), "B": (6, 0.0005, 256), "C": (6, 0.0012, 256)}


def parent_checkpoint(rnd):
    weights = [path for path in rnd.parent.descriptor["checkpoint_paths"] if path.endswith(".npz")]
    if len(weights) != 1:
        raise ValueError("父产物应声明一个工业模型 .npz 检查点")
    path = rnd.parent.path(weights[0])
    digest = checkpoint_digest(path)
    if digest.removeprefix("sha256:") != rnd.parent.checkpoint_sha256.removeprefix("sha256:"):
        raise ValueError("父检查点与平台本轮指定的摘要不符")
    return path, digest


def propose(rnd, history, parent, public_parent, rows):
    context = {
        "round": rnd.round_index, "training_rows": len(rows["ids"]),
        "parent": {key: parent.meta.get(key) for key in ("epochs", "learning_rate", "batch_size")},
        "parent_public_validation": public_parent,
        "history": [{key: item.get(key) for key in ("id", "configuration", "reason", "change_summary")}
                    for item in history[-3:]],
        "decision_code": inspect.getsource(accept_candidate),
    }
    prompt = (
        "读下面的历史、当前状态和选模代码，提出一次 CPU 续训实验。只输出一行 Python 策略声明，"
        '例如 experiment(name="small_update", data="full", epochs=3, '
        'learning_rate=0.001, batch_size=128, reason="限制更新幅度")。'
        "data 固定为 full（本轮授权训练投影），epochs 为 1 到 8，学习率 1e-5 到 3e-3，"
        "batch_size 为 64、128 或 256，说明如何参考历史或父模型。\n"
        + json.dumps(context, ensure_ascii=False, allow_nan=False)
    )
    receipt = rnd.model([{"role": "user", "content": prompt}],
                        max_tokens=min(1024, rnd.max_output_tokens), temperature=0, enable_thinking=False)
    source = receipt.text.strip()
    if source.startswith("```python\n") and source.endswith("\n```"):
        source = source[len("```python\n"):-len("\n```")].strip()
    elif source.startswith("```\n") and source.endswith("\n```"):
        source = source[len("```\n"):-len("\n```")].strip()
    try:
        proposal, = parse_strategy(source, data_strategies=("full",))
        return proposal, source, "pytrio"
    except ValueError as error:
        epochs, rate, batch = FALLBACK[rnd.dataset_key]
        fallback = dict(name="conservative", data="full", epochs=epochs, learning_rate=rate, batch_size=batch,
                        reason="模型回复无法解析，执行保守续训方案。")
        source = "experiment(" + ", ".join(f"{key}={value!r}" for key, value in fallback.items()) + ")"
        print(f"round {rnd.round_index}: {fallback['reason']} {error}", file=sys.stderr)
        return fallback, source, "rules_after_model_reply"


def public_metric(summary, metric):
    if not isinstance(summary, dict):
        return None
    values = summary.get("metrics", summary)
    value = values.get(metric) if isinstance(values, dict) else None
    if type(value) in (int, float) and math.isfinite(value):
        return value
    return None


def accept_candidate(metric, parent, candidate):
    if metric == "macro_f1":
        return candidate >= parent + 0.003
    return candidate < parent and candidate <= parent * 0.998


def train(rnd):
    task = rnd.task["task"]
    metric = rnd.task["metric"]
    profile = ROOT / "configs" / f"{rnd.dataset_key}.json"
    expected = json.loads(profile.read_text(encoding="utf-8"))["task"]
    if (task, metric) != (expected, "macro_f1" if expected == "classification" else "rmse"):
        raise ValueError("平台任务或指标与工业模型配置不一致")

    parent_path, parent_digest = parent_checkpoint(rnd)
    parent = Model.load(parent_path)
    if (parent.meta["profile"], parent.meta["task"]) != (rnd.dataset_key, task):
        raise ValueError("父检查点的任务与本轮数据集不一致")
    classes = parent.meta["classes"]
    train_rows = read_dataset(rnd.data.train_path, task, classes, parent.meta["paths"])
    validation = read_dataset(rnd.data.validation_path, task, classes, parent.meta["paths"])
    parent_validation = metrics(parent, validation)
    history = rnd.history()
    plan, source, plan_source = propose(rnd, history, parent, parent_validation, train_rows)
    changes = {field: plan[field] for field in ("epochs", "learning_rate", "batch_size")}
    hypothesis = plan["reason"]
    summary = (f"从平台父检查点继续训练 {changes['epochs']} 个 epoch，"
               f"学习率 {changes['learning_rate']}，批次大小 {changes['batch_size']}。")
    configuration = {**changes, "data": "public_train", "plan_source": plan_source, "candidate": plan["name"]}
    seed = parent.meta["seed"]

    with TemporaryDirectory(prefix="sdk-round-", dir=os.getenv("CONTEST_WORK", "/workspace")) as directory:
        candidate = Path(directory) / "candidate"
        report = train_run(profile, rnd.data.train_path, rnd.data.validation_path, candidate,
                           resume=parent_path, seed=seed, stage_id=f"{rnd.dataset_key}-round-{rnd.round_index}-{plan['name']}",
                           prepared_train=train_rows, prepared_validation=validation, **changes)
        if report["parent_checkpoint_sha256"] != parent_digest or report["final_steps"] <= report["initial_steps"]:
            raise RuntimeError("候选必须是本轮父模型的真实续训产物")
        (candidate / "strategy.py").write_text(source + "\n", encoding="utf-8")
        submitted = rnd.submit_candidate(
            candidate, hypothesis=hypothesis, change_summary=summary, configuration=configuration,
            seed=seed, checkpoint_paths=["checkpoint.npz"], code_paths=["strategy.py"], predict_argv=PREDICT_ARGV,
        )
        before = public_metric(rnd.frame.get("current_validation"), metric)
        after = public_metric(submitted.get("summary"), metric)
        comparison = "平台公开验证指标"
        if before is None or after is None:
            before = report["validation_before"][metric]
            after = report["validation_after"][metric]
            comparison = "相同公开验证投影上的本地成对比较"
        adopted = accept_candidate(metric, before, after)
        selected = submitted["artifact_id"] if adopted else rnd.parent.artifact_id
        decision = "达到改进阈值，采用续训候选" if adopted else "未达到改进阈值，保留父模型"
        rnd.finish(
            selected_artifact_id=selected, hypothesis=hypothesis, change_summary=summary,
            reason=f"{comparison}的 {metric} {decision}。本轮计划来源 {plan_source}。",
            configuration=configuration, seed=seed,
            history_ids=[item["id"] for item in history], model_call_ids=list(rnd.call_ids),
        )


@lru_cache(maxsize=1)
def load_prediction_model(path):
    return Model.load(path)


def predict(sample, model_dir):
    root = model_dir.resolve()
    path = Path(os.getenv("CONTEST_ARTIFACT_PATH", "checkpoint.npz"))
    path = path if path.is_absolute() else root / path
    if not path.resolve().is_relative_to(root):
        raise ValueError("预测检查点必须位于平台挂载的 /model 内")
    model = load_prediction_model(path)
    if model.meta["task"] != sample.task or sample.dataset_key not in (None, model.meta["profile"]):
        raise ValueError("预测请求与平台选中的检查点不匹配")
    result = predict_checkpoint(model, sample.inputs)
    if sample.task == "classification":
        scores = result["outputs"]
        best = max(item["score"] for item in scores)
        tied = [item for item in scores if item["score"] == best]
        for item in tied[1:]:
            item["score"] = math.nextafter(best, 0.0)
    return result
