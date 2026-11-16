# -*- coding: utf-8 -*-
"""将已完成的 Harness 本地指标补传到指定 SwanLab 项目。"""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import swanlab


def load_run(directory):
    raw_log = (directory / "engine_log.jsonl").read_bytes()
    events = [json.loads(line) for line in raw_log.splitlines() if line.strip()]
    if not events or events[0]["type"] != "run_start" or events[-1]["type"] != "run_end":
        raise ValueError("需要包含 run_start 和 run_end 的完整运行日志")
    start, summary = events[0], events[-1]["summary"]
    if not summary["rounds"] or start["task"] != summary["task"]:
        raise ValueError("运行摘要不完整或任务不匹配")
    metadata = {kind: {event["round"]: event for event in events if event["type"] == kind}
                for kind in ("diagnosis", "plan")}
    metadata["completed"] = [event for event in events if event["type"] == "experiment"
                             and event.get("status") == "completed"]
    return start, summary, metadata, hashlib.sha256(raw_log).hexdigest()


def round_metrics(row, metadata, task):
    round_no = row["round"]
    diagnosis = metadata["diagnosis"].get(round_no, {}).get("diagnosis", {})
    plan = metadata["plan"].get(round_no, {})
    source = row.get("plan_source", plan.get("source", "rules"))
    metric = "macro_f1" if task == "A" else "rmse"
    values = {
        f"round/{metric}_baseline": row["static_primary"],
        f"round/{metric}_before": row["adopted_zero_primary"],
        f"round/{metric}_after": row["adopted_after_primary"],
        "round/experiments": row["n_experiments"],
        "round/used_pytrio": int(source in ("pytrio", "hybrid")),
        "round/decision": {"DEPLOY_NEW": 1, "KEEP": 0, "ROLLBACK": -1}[row["decision"]],
        "drift/max_psi": diagnosis.get("max_psi"),
        "drift/max_mean_shift": diagnosis.get("max_abs_shift"),
    }
    for slot, experiment in enumerate(row["experiments"], 1):
        for field in ("primary_val", "replay_val", "seconds", "epochs", "learning_rate"):
            values[f"experiment_{slot}/{field}"] = experiment.get(field)
    metrics = {key: value for key, value in values.items()
               if isinstance(value, (int, float)) and math.isfinite(value)}
    details = {"决策": row["decision"], "计划来源": source, "选中候选": row["chosen_experiment"],
               "漂移诊断": row["diagnosis"], "候选": row["experiments"]}
    metrics["round/details"] = swanlab.Text(json.dumps(details, ensure_ascii=False, indent=2))
    return metrics


def epoch_metrics(directory, completed, task):
    histories = {path.parent.name: path for path in directory.glob("round*-exp*/history.jsonl")}
    points = defaultdict(dict)
    slots = defaultdict(int)
    metric = "macro_f1" if task == "A" else "rmse"
    for event in completed:
        round_no = event["round"]
        slots[round_no] += 1
        path = histories.get(event["candidate"]["stage_id"])
        if path is None:
            continue
        key = f"train_round_{round_no:02d}/experiment_{slots[round_no]}"
        with path.open(encoding="utf-8") as stream:
            for epoch, line in enumerate((line for line in stream if line.strip()), 1):
                record = json.loads(line)
                values = {f"{key}_loss": record.get("loss"),
                          f"{key}_val_{metric}": record.get("validation", {}).get(metric)}
                points[epoch].update({name: value for name, value in values.items()
                                      if isinstance(value, (int, float)) and math.isfinite(value)})
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="包含 engine_log.jsonl 的目录")
    parser.add_argument("--workspace", required=True, help="SwanLab 目标空间")
    parser.add_argument("--project", default="smart-training-harness")
    parser.add_argument("--name", required=True, help="用于区分训练、联调和容器冒烟的名称")
    parser.add_argument("--epochs-only", action="store_true", help="仅为已补传的实验追加 epoch 曲线")
    args = parser.parse_args()
    start, summary, metadata, digest = load_run(args.run)
    task = summary["task"]
    epochs = epoch_metrics(args.run, metadata["completed"], task)
    if args.epochs_only and not any(epochs.values()):
        parser.error("运行目录缺少候选实验的逐 epoch 记录 history.jsonl")
    settings = swanlab.Settings(interactive=False, probe={
        "hardware": False, "runtime": False, "requirements": False,
        "git": False, "monitor": False,
    })
    with swanlab.init(
        workspace=args.workspace, project=args.project, name=args.name,
        mode="online", on_online_error="raise", settings=settings,
        log_dir=str(args.run / "swanlab_backfill"),
        id=digest[:12], resume="must" if args.epochs_only else "allow", job_type="harness-backfill",
        description="从已完成的 Harness 日志补传逐轮过程和结果。",
        config={"task": task, "rounds": summary["decision_count"],
                "experiments": summary["experiment_count"], "pytrio": summary["llm_used"],
                "model": start.get("llm", {}).get("model"), "origin": "local-engine-log",
                "original_start": datetime.fromtimestamp(start["ts"]).astimezone().isoformat()},
    ) as run:
        url = run.url
        if not url:
            raise RuntimeError("SwanLab 未返回云端实验地址")
        if not args.epochs_only:
            for row in summary["rounds"]:
                run.log(round_metrics(row, metadata, task), step=row["round"])
            final_metrics = {f"final/{stage}_{key}": value
                             for stage, metrics in summary["final_replay"].items()
                             for key, value in metrics.items()
                             if isinstance(value, (int, float)) and math.isfinite(value)}
            run.log(final_metrics, step=summary["decision_count"])
        for epoch, metrics in sorted(epochs.items()):
            if metrics:
                run.log(metrics, step=epoch)
    print(url)


if __name__ == "__main__":
    main()
