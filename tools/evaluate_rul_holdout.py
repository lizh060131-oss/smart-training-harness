# -*- coding: utf-8 -*-
"""在官方公开的整设备留出集上比较 B/C 续训检查点与官方起点。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "1" / "harness_v2"
PUBLIC_PACK = {"B": "cpu-b-rul-20261110-v1", "C": "cpu-c-guide-20261110-v1"}
sys.path.insert(0, str(HARNESS))
from baselines.data import read_dataset  # noqa: E402
from baselines.model import Model, checkpoint_digest  # noqa: E402


def devices(path, task):
    """用数据包元数据定义设备身份，同一台发动机的全部循环共用一个键。"""
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("meta", {})
            keys = ("bearing",) if task == "B" else ("source_subset", "unit")
            if incomplete := [key for key in keys if meta.get(key) in (None, "")]:
                raise ValueError(f"{path} 第 {number} 行缺少设备字段 {', '.join(incomplete)}")
            yield row["id"], tuple(str(meta[key]) for key in keys)


def errors(residual):
    return {"rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual)))}


def evaluate(model, dataset, groups, task):
    residual = model.predict(dataset["x"]) - dataset["y"]
    by_device = {device: errors(residual[indices]) for device, indices in groups.items()}
    rmses = [result["rmse"] for result in by_device.values()]
    report = {**errors(residual), "device_mean_rmse": float(np.mean(rmses)),
              "worst_device_rmse": max(rmses)}
    if task == "B":
        report["by_bearing"] = {
            device[0]: {"rows": len(groups[device]), **by_device[device]}
            for device in sorted(groups)
        }
    else:
        by_subset = defaultdict(list)
        for (subset, _), indices in groups.items():
            by_subset[subset].extend(indices)
        report["by_subset"] = {subset: {"rows": len(indices), **errors(residual[indices])}
                               for subset, indices in sorted(by_subset.items())}
        # NASA C-MAPSS/PHM08 的低估与高估误差尺度是 13 与 10。
        # 本工具在公开留出集上逐行求和，固定同一数据集才能横比。
        # https://ntrs.nasa.gov/api/citations/20205001125/downloads/Run_to_Failure_Simulation_Under_Real_Flight_Conditions_Dataset.pdf
        penalty = np.where(residual < 0, -residual / 13, residual / 10)
        report["phm08_sample_sum"] = float(np.expm1(penalty).sum())
    return report


def evaluate_run(task, run, round_no, data_dir):
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary["task"] != task:
        raise ValueError(f"运行任务为 {summary['task']}，与 --task {task} 不符")
    rounds = summary["rounds"][:round_no]
    if [row["round"] for row in rounds] != list(range(1, round_no + 1)):
        raise ValueError(f"运行摘要未包含第 1–{round_no} 轮完整记录")

    pack = ROOT / "data" / "official_baselines" / PUBLIC_PACK[task]
    baseline = pack / "checkpoint.npz"
    with (run / "engine_log.jsonl").open(encoding="utf-8") as stream:
        start = json.loads(stream.readline())
    if start.get("type") != "run_start" or start.get("baseline") != checkpoint_digest(baseline):
        raise ValueError("运行日志中的起点与当前官方基线检查点不一致")

    holdout = pack / "public" / "validation.jsonl"
    device_by_id = dict(devices(holdout, task))
    withheld = set(device_by_id.values())
    for record in rounds:
        batch = Path(record["batch_file"].replace("\\", "/"))
        batch = data_dir / batch.name if data_dir else ROOT / batch
        if not batch.is_file():
            raise ValueError(f"找不到第 {record['round']} 轮原始批次 {batch}，可用 --data-dir 指定原数据目录")
        overlap = withheld.intersection(device for _, device in devices(batch, task))
        if overlap:
            example = ", ".join("/".join(key) for key in sorted(overlap)[:3])
            raise ValueError(f"第 {record['round']} 轮使用过 {len(overlap)} 个留出设备（如 {example}），请选更早轮次")

    stage = rounds[-1]["adopted_stage"]
    if not stage or stage in (".", "..") or any(separator in stage for separator in ("/", "\\", ":")):
        raise ValueError("运行摘要的检查点阶段无效")
    checkpoint = baseline if stage == "official_baseline" else run / stage / "checkpoint.npz"
    original = Model.load(baseline)
    adopted = Model.load(checkpoint)
    if adopted.meta["profile"] != task or adopted.meta["paths"] != original.meta["paths"]:
        raise ValueError("选中检查点与官方任务或特征协议不一致")
    dataset = read_dataset(holdout, "regression", expected_paths=original.meta["paths"])
    groups = defaultdict(list)
    for index, identity in enumerate(dataset["ids"]):
        groups[device_by_id[identity]].append(index)
    return {"task": task, "round": round_no, "adopted_stage": stage,
            "adopted_digest": checkpoint_digest(checkpoint), "holdout_digest": dataset["sha256"],
            "holdout_rows": len(dataset["ids"]), "holdout_devices": len(groups),
            "official_baseline": evaluate(original, dataset, groups, task),
            "adopted": evaluate(adopted, dataset, groups, task)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("B", "C"), required=True)
    parser.add_argument("--run", type=Path, required=True, help="包含 summary.json 的本地运行目录")
    parser.add_argument("--round", type=int, default=9, help="检查点采用轮次，默认 9")
    parser.add_argument("--data-dir", type=Path, help="旧日志路径无法访问时，指定原始批次目录")
    args = parser.parse_args()
    if args.round < 1:
        parser.error("--round 必须至少为 1")
    try:
        report = evaluate_run(args.task, args.run, args.round, args.data_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        sys.exit(f"评估失败：{exc}")
