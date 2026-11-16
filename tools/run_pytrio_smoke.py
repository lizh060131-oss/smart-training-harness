# -*- coding: utf-8 -*-
"""使用真实 PyTRiO 推理接口联调 A 任务的前四轮。"""
import argparse
from datetime import datetime
from getpass import getpass
import json
import os
from pathlib import Path
import subprocess
import sys

from build_development_streams import write_stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("PETRIO_MODEL"),
                        help="赛事允许的完整模型 ID")
    parser.add_argument("--base-url", default=os.environ.get("PETRIO_BASE_URL") or
                        "https://pytrio.com/api/openai/v1",
                        help="OpenAI 兼容 API 的基础地址，以 /v1 结尾")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    if not base_url.startswith("https://") or not base_url.endswith("/v1"):
        parser.error("--base-url 应使用 PyTRiO 控制台给出的 HTTPS /v1 地址")

    try:
        model = (args.model or input("粘贴赛题允许的完整模型 ID：")).strip()
        if not model:
            parser.error("模型 ID 不能为空")
        key = os.environ.get("PETRIO_API_KEY", "").strip()
        if not key:
            if not sys.stdin.isatty():
                parser.error("请在本机交互式终端运行，以隐藏输入 API Key")
            key = getpass("粘贴 PyTRiO API Key（输入时屏幕不会变化，粘贴后按回车）：").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n已取消输入，尚未向 PyTRiO 发出请求。", file=sys.stderr)
        return 130
    if not key:
        parser.error("API Key 不能为空")

    root = Path(__file__).resolve().parents[1]
    out = root / "1" / "runs_v2" / datetime.now().strftime("gas_pytrio_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    stream = out / "development_stream"
    write_stream("A", stream)
    rounds = 4
    env = dict(os.environ, PETRIO_BASE_URL=base_url, PETRIO_MODEL=model,
               PETRIO_API_KEY=key, PYTHONUTF8="1")
    print(f"输出目录 {out}", flush=True)
    run = subprocess.run(
        [sys.executable, str(root / "1" / "harness_v2" / "main.py"),
         "--task", "A", "--data", str(stream),
         "--rounds", str(rounds), "--out", str(out)], cwd=root, env=env)
    if run.returncode != 0:
        return run.returncode
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if summary["llm_rounds"] != rounds:
        print(f"PyTRiO 计划只用于 {summary['llm_rounds']}/{rounds} 轮。"
              f"请查阅 {out / 'engine_log.jsonl'} 中的 llm_error。", file=sys.stderr)
        return 1
    tool_rounds = [row["round"] for row in summary["rounds"] if row["round"] > 1
                   and row["plan_source"] == "hybrid" and row.get("tool_protocol") == "policy_dsl"]
    if len(tool_rounds) != rounds - 1:
        print(f"模型已规划 {rounds} 轮，其中 {len(tool_rounds)} 轮完成了带历史的工具流程。"
              f"请核对 {out / 'engine_log.jsonl'} 中的 model_tool 事件。", file=sys.stderr)
        return 1
    print(f"联调通过：{rounds}/{rounds} 轮使用模型计划，第 2–{rounds} 轮完成带历史的工具流程。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
