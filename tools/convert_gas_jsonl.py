# -*- coding: utf-8 -*-
"""Convert official gas batchN.dat (libsvm) -> platform JSONL format.

id scheme mirrors the official package: gas_b{NN}_{index:06d}
inputs.features = 128 floats in feature-index order (1..128)
answer = English class name; meta = {source_batch, source_index}
Only finite values kept (official converter keeps all finite features).
"""
import json
import os
import sys

CLASSES = {1: "Ethanol", 2: "Ethylene", 3: "Ammonia",
           4: "Acetaldehyde", 5: "Acetone", 6: "Toluene"}


def parse_libsvm(line):
    parts = line.strip().split()
    if not parts:
        return None
    label = int(parts[0])
    feats = [None] * 128
    for token in parts[1:]:
        idx, val = token.split(":", 1)
        idx = int(idx)
        if 1 <= idx <= 128:
            feats[idx - 1] = float(val)
    if any(v is None for v in feats):
        raise ValueError("line missing features: %r" % line[:80])
    return label, feats


def convert(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows_total = 0
    for batch in range(1, 11):
        src = os.path.join(src_dir, "batch%d.dat" % batch)
        dst = os.path.join(out_dir, "batch%02d.jsonl" % batch)
        with open(src, "r", encoding="utf-8") as fin, \
             open(dst, "w", encoding="utf-8", newline="\n") as fout:
            n = 0
            for line in fin:
                parsed = parse_libsvm(line)
                if parsed is None:
                    continue
                label, feats = parsed
                n += 1
                row = {
                    "id": "gas_b%02d_%06d" % (batch, n),
                    "inputs": {"features": feats},
                    "answer": CLASSES[label],
                    "meta": {"source_batch": batch, "source_index": n},
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_total += n
            print("batch%02d: %d rows -> %s" % (batch, n, dst))
    print("TOTAL %d rows" % rows_total)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
