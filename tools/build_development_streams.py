# -*- coding: utf-8 -*-
"""构建不含官方公开验证集的三项本地十轮验证流。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKS = {"A": "cpu-a-guide-20261110-v1", "B": "cpu-b-rul-20261110-v1",
         "C": "cpu-c-guide-20261110-v1"}


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def c_stream(rows):
    subsets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        meta = row["meta"]
        subsets[meta["source_subset"]][meta["unit"]].append(row)
    batches = [[] for _ in range(10)]
    for subset in sorted(subsets):
        for i, (_, trajectory) in enumerate(sorted(subsets[subset].items())):
            batches[i % len(batches)].extend(trajectory)
    return batches


def make_batches(task, split, held_out):
    if task == "A":
        validation_batches = {row["meta"]["source_batch"] for row in held_out}
        return [batch for path in sorted((ROOT / "data/gas_jsonl").glob("batch*.jsonl"))
                if (batch := read_rows(path)) and batch[0]["meta"]["source_batch"] not in validation_batches]
    rows = read_rows(split / "train.jsonl")
    if task == "C":
        return c_stream(rows)
    bearings = defaultdict(list)
    for row in rows:
        bearings[row["meta"]["bearing"]].append(row)
    return list(bearings.values())


def write_stream(task, destination):
    split = ROOT / "data/official_baselines" / PACKS[task] / "public"
    held_out = read_rows(split / "validation.jsonl")
    batches = make_batches(task, split, held_out)
    if len(batches) == 9:
        # 本地样本仅有九个原始阶段，将最大的阶段按原始行顺序切成两次释放。
        largest = max(range(len(batches)), key=lambda i: len(batches[i]))
        batch = batches[largest]
        midpoint = len(batch) // 2
        batches[largest:largest + 1] = [batch[:midpoint], batch[midpoint:]]
    if len(batches) != 10 or any(len(batch) < 3 for batch in batches):
        raise ValueError(f"任务 {task} 无法构造十个非空批次")

    if task == "B":
        def group(row):
            return row["meta"]["bearing"]
    elif task == "C":
        def group(row):
            return (row["meta"]["source_subset"], row["meta"]["unit"])
    else:
        def group(row):
            return row["id"]
    withheld = {group(row) for row in held_out}
    if any(group(row) in withheld for batch in batches for row in batch):
        raise ValueError(f"任务 {task} 的训练流与官方验证重叠")
    destination.mkdir()
    for i, batch in enumerate(batches, 1):
        with (destination / f"batch{i:02d}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in batch:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(task, [len(batch) for batch in batches])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="新建的输出目录")
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    for task in "ABC":
        write_stream(task, args.out / task)
