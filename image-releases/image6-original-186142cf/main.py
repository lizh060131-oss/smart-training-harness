# -*- coding: utf-8 -*-
"""智训引擎 v2 统一入口。

用法（示例）：
  python main.py --task A --data data/gas_jsonl --out 1/runs_v2/gas
  python main.py --task B --data data/bearing_jsonl --out 1/runs_v2/bearing
  python main.py --task C --data data/cmapss_jsonl --out 1/runs_v2/cmapss

环境变量（可选，LLM 顾问）：
  PETRIO_API_KEY / PETRIO_BASE_URL / PETRIO_MODEL（OpenAI 兼容 PyTRIO 服务）
未配置时使用规则规划器，运行记录中如实标注 llm_used=false。
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import RoundEngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="智训引擎 v2 统一入口")
    ap.add_argument("--task", required=True, choices=["A", "B", "C"])
    ap.add_argument("--data", required=True, help="目录，含 batchNN.jsonl（按批次顺序）")
    ap.add_argument("--out", required=True, help="输出目录（每轮实验/检查点/日志）")
    ap.add_argument("--baseline", default=None,
                    help="官方基线检查点；默认 harness_v2/checkpoints/{TASK}-baseline.npz")
    ap.add_argument("--rounds", type=int, default=None, help="训练轮数，默认最多 10 轮")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--replay-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-experiments", type=int, default=3)
    args = ap.parse_args()

    task = args.task
    data_dir = Path(args.data)
    batch_files = sorted(data_dir.glob("batch*.jsonl"))
    if not batch_files:
        sys.exit("no batchNN.jsonl found under %s" % data_dir)
    if args.rounds is not None and not 1 <= args.rounds <= len(batch_files):
        ap.error("--rounds must be between 1 and %d" % len(batch_files))
    rounds = args.rounds if args.rounds is not None else min(10, len(batch_files))
    profile = ROOT / "configs" / ("%s.json" % task)
    baseline = Path(args.baseline) if args.baseline else ROOT / "checkpoints" / ("%s-baseline.npz" % task)
    if not baseline.is_file():
        sys.exit("baseline checkpoint not found: %s" % baseline)

    engine = RoundEngine(
        task=task, profile_path=str(profile), baseline_ckpt=str(baseline),
        batch_files=batch_files, out_dir=args.out,
        rounds=rounds, val_ratio=args.val_ratio,
        replay_ratio=args.replay_ratio, seed=args.seed,
        min_experiments=args.min_experiments)
    summary = engine.run()

    print("\n===== 最终结果 =====")
    print("采纳的最终检查点:", summary["adopted_final_stage"])
    for r in summary["rounds"]:
        print("round %02d | %-10s | static %.4f | adopted_zero %.4f | adopted_after %.4f | gain %+.4f | %s" % (
            r["round"], r["decision"], r["static_primary"],
            r["adopted_zero_primary"], r["adopted_after_primary"],
            r["gain_vs_static"], r["chosen_experiment"] or "-"))
    print("最终 replay 指标:", json.dumps(summary["final_replay"], ensure_ascii=False))
    print("实验总数 %d，决策轮数 %d，LLM=%s" % (
        summary["experiment_count"], summary["decision_count"], summary["llm_used"]))


if __name__ == "__main__":
    main()
