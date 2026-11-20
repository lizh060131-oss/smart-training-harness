"""C 的正常数据分组门控；验证答案只用于选模，不参与训练。"""
import hashlib
import json
import re
import shutil

import numpy as np
from baselines.data import read_dataset
from baselines.model import Model
from baselines.train import metrics


def subset(sample_id):
    match = re.match(r"(fd00[1-4])_", sample_id.lower())
    return match.group(1) if match else "unidentified"


def grouped_metrics(model, data, prefix):
    result = {prefix + "/all": metrics(model, data)}
    groups = {}
    for index, identifier in enumerate(data["ids"]):
        groups.setdefault(subset(identifier), []).append(index)
    for name, indices in groups.items():
        selected = np.asarray(indices)
        result[prefix + "/" + name] = metrics(model, {
            **data, "x": data["x"][selected], "y": data["y"][selected]})
    return result


def eligible(parent, reference, candidate):
    # 与本轮父模型及最初参考模型逐组比较，不容许混合指标抵消退步。
    for name, values in candidate.items():
        for metric in ("rmse", "mae"):
            value = values[metric]
            if not np.isfinite(value) or value > min(parent[name][metric], reference[name][metric]) + 1e-12:
                return False
    return candidate["current/all"]["rmse"] <= parent["current/all"]["rmse"] * .998


def review(rnd, root, parent_path, trials, default_best):
    state = rnd.state_workspace(root / "state")
    reference_path = state / "reference.npz"
    if not reference_path.exists():
        shutil.copyfile(parent_path, reference_path)
    parent, reference = Model.load(parent_path), Model.load(reference_path)
    rows = list(rnd.data.validation_rows())
    current_ids = {row["id"] for row in rows}
    history_path = state / "validation.json"
    old = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    datasets = {"current": read_dataset(rnd.data.validation_path, "regression", [], parent.meta["paths"])}
    for index, group in enumerate(old):
        remaining = [row for row in group if row["id"] not in current_ids]
        if remaining:
            path = root / f"history-{index}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in remaining), encoding="utf-8")
            datasets[f"history-{index}"] = read_dataset(path, "regression", [], parent.meta["paths"])

    def evaluate(model):
        result = {}
        for label, data in datasets.items():
            result.update(grouped_metrics(model, data, label))
        return result

    before, anchor = evaluate(parent), evaluate(reference)
    accepted = []
    for index, trial in enumerate(trials):
        after = evaluate(Model.load(root / str(index) / "checkpoint.npz"))
        trial["c_group_metrics"] = after
        trial["c_gate_passed"] = eligible(before, anchor, after)
        if trial["c_gate_passed"]:
            accepted.append(index)
    best = min(accepted, key=lambda i: trials[i]["validation_after"]["rmse"]) if accepted else default_best
    groups = {}
    for row in rows:
        groups.setdefault(subset(row["id"]), []).append({k: row[k] for k in ("id", "inputs", "answer")})
    sampled = [row for group in groups.values()
               for row in sorted(group, key=lambda r: hashlib.sha256(r["id"].encode()).digest())[:100]]
    history_path.write_text(json.dumps((old + [sampled])[-10:]), encoding="utf-8")
    audit = {"parent": before, "reference": anchor, "groups": list(before),
             "required_current_rmse_improvement": .002,
             "rule": "Every available group RMSE and MAE must not exceed parent or initial reference; current RMSE must improve by at least 0.2%.",
             "accepted_trials": accepted}
    return best, bool(accepted), audit, state
