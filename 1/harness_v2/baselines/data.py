"""Read only explicitly supplied public training/validation data."""
import hashlib
from functools import lru_cache
import json
import math
from pathlib import Path
import re

import numpy as np

MAX_BYTES = 128 * 1024 * 1024
FORBIDDEN_INPUTS = {"truerul", "normalizedrul", "remainingusefullife", "rul", "totalsnapshots"}


def sha256(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def json_value(raw):
    def reject(_):
        raise ValueError("nonfinite JSON number")
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject)


@lru_cache(maxsize=8192)
def path_json(path):
    return json.dumps(path, ensure_ascii=False, separators=(",", ":"))


@lru_cache(maxsize=512)
def forbidden_key(key):
    return re.sub(r"[^a-z0-9]", "", key.lower()) in FORBIDDEN_INPUTS


def flatten(inputs):
    paths, values = [], []
    def visit(value, path, depth):
        if depth > 16:
            raise ValueError("input nesting exceeds limit")
        if isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str) or forbidden_key(key):
                    raise ValueError("target-derived input field is not allowed")
                visit(value[key], path + (("key", key),), depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + (("index", index),), depth + 1)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            paths.append(path_json(path))
            values.append(float(value))
        else:
            raise ValueError("model inputs must contain finite numeric features only")
    visit(inputs, (), 0)
    if not values or len(values) > 4096:
        raise ValueError("expected 1..4096 numeric features")
    return paths, values


def parse_target(answer, task, classes):
    if task == "classification":
        if not isinstance(answer, str) or not classes or answer not in classes:
            raise ValueError("label is outside the configured class list")
        return classes.index(answer)
    if task == "regression":
        if isinstance(answer, bool) or not isinstance(answer, (float, int)) or not math.isfinite(answer) or answer < 0:
            raise ValueError("RUL must be a finite nonnegative number")
        return float(answer)
    raise ValueError("unsupported task")


def check_dataset_file(path):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_BYTES:
        raise ValueError("dataset must be a regular JSONL file no larger than 128 MiB")


def dataset_digest(path):
    path = Path(path)
    check_dataset_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


class PreparedRows:
    """Write the audited JSONL and reuse the exact same validated feature rows across candidates."""

    def __init__(self, task, classes):
        self.task = task
        self.classes = classes
        self.paths = None
        self.features = {}

    def vector(self, inputs):
        key = id(inputs)
        if key not in self.features:
            paths, values = flatten(inputs)
            if self.paths is None:
                self.paths = paths
            if paths != self.paths:
                raise ValueError("feature schema changed")
            # Retain the object: it cannot be changed during training or recycled into the same id.
            self.features[key] = (inputs, values)
        return self.features[key][1]

    def write(self, path, rows):
        ids, features, targets, seen = [], [], [], set()
        digest = hashlib.sha256()
        size = 0
        with Path(path).open("wb") as stream:
            for row in rows:
                identity = row["id"]
                if not isinstance(identity, str) or not identity or identity in seen:
                    raise ValueError("sample IDs must be nonempty and unique")
                features.append(self.vector(row["inputs"]))
                targets.append(parse_target(row["answer"], self.task, self.classes))
                blob = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                size += len(blob)
                if size > MAX_BYTES:
                    raise ValueError("dataset exceeds 128 MiB")
                stream.write(blob)
                digest.update(blob)
                ids.append(identity)
                seen.add(identity)
        if not ids:
            raise ValueError("dataset is empty")
        return {"ids": ids, "x": np.asarray(features, dtype=np.float64),
                "y": np.asarray(targets, dtype=np.int64 if self.task == "classification" else np.float64),
                "paths": self.paths, "sha256": "sha256:" + digest.hexdigest()}


def read_dataset(path, task, classes=None, expected_paths=None):
    path = Path(path)
    check_dataset_file(path)
    raw = path.read_bytes()
    ids, features, targets = [], [], []
    seen = set()
    paths = expected_paths
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json_value(line)
        if not isinstance(row, dict) or not {"id", "inputs", "answer"} <= set(row):
            raise ValueError(f"line {number}: id, inputs and answer are required")
        identity = row["id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError(f"line {number}: sample IDs must be nonempty and unique")
        current_paths, values = flatten(row["inputs"])
        if paths is None:
            paths = current_paths
        if current_paths != paths:
            raise ValueError(f"line {number}: feature schema changed")
        try:
            target = parse_target(row["answer"], task, classes)
        except ValueError as exc:
            raise ValueError(f"line {number}: {exc}") from exc
        ids.append(identity); seen.add(identity); features.append(values); targets.append(target)
    if not ids:
        raise ValueError("dataset is empty")
    return {"ids": ids, "x": np.asarray(features, dtype=np.float64),
            "y": np.asarray(targets, dtype=np.int64 if task == "classification" else np.float64),
            "paths": paths, "sha256": sha256(raw)}
