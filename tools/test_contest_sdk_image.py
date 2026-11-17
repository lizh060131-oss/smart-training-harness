"""Drive an SDK contest image against a local host using the public baseline packages."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "contest.harness.workspace.v2"
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
PACKS = {"A": "cpu-a-guide-20261110-v1", "B": "cpu-b-rul-20261110-v1", "C": "cpu-c-guide-20261110-v1"}


def docker(image, command, *, mounts=(), environment=(), interactive=False):
    args = ["docker", "run", "--rm", "--platform", "linux/amd64", "--network", "none",
            "--read-only", "--cpus", "4", "--memory", "4g"]
    if interactive:
        args += ["-i"]
    for path, target in mounts:
        args += ["--mount", f"type=bind,source={path},target={target},readonly"]
    for name, value in environment:
        args += ["-e", f"{name}={value}"]
    if command == "train":
        args += ["--tmpfs", "/workspace:rw,noexec,nosuid,size=128m"]
    return [*args, image, command]


def emit(process, frame):
    process.stdin.write(json.dumps(frame, ensure_ascii=False) + "\n")
    process.stdin.flush()


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def round_trip(image, task, directory, public, parent, parent_id, round_index, history, *, promote):
    parent_raw = (parent / "checkpoint.npz").read_bytes()
    parent_digest = digest(parent_raw)
    kind = "classification" if task == "A" else "regression"
    metric = "macro_f1" if task == "A" else "rmse"
    if task == "A":
        before, after = 0.70, 0.73 if promote else 0.67
    else:
        before, after = 60.0, 58.2 if promote else 61.8
    parent_receipt = {
        "artifact_id": parent_id,
        "artifact": {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [], "predict_argv": PREDICT_ARGV},
        "files": [{"path": "checkpoint.npz", "sha256": parent_digest, "size_bytes": len(parent_raw)}],
    }
    prompt = "HOST_ONLY: official prompt inserted by the host"
    start = {
        "type": "start", "protocol": PROTOCOL, "prompt": prompt,
        "prompt_version": "industrial-workspace-flexible-v3", "dataset_key": task, "round_index": round_index,
        "task_contract": {"task": kind, "metric": metric, "metric_direction": "higher" if task == "A" else "lower"},
        "model_name": "base:local-protocol-fixture", "max_output_tokens": 1024,
        "parent_checkpoint_sha256": parent_digest, "parent_artifact": parent_receipt, "artifacts": [parent_receipt],
        "current_validation": {"metrics": {metric: before}},
    }
    command = docker(image, "train", mounts=((public, "/data"), (parent, "/previous")), interactive=True)
    candidate_id = f"local-artifact-{task}-{round_index}"
    call_id = f"local-model-call-{task}-{round_index}"
    candidate = directory / candidate_id
    invalid_reply = task == "B"
    with (directory / f"{task}-{round_index}.stderr").open("w+", encoding="utf-8") as stderr:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                              encoding="utf-8") as process:
            try:
                emit(process, start)
                for sequence, name in enumerate(("history_request", "model_request", "candidate_request", "round_complete")):
                    frame = json.loads(process.stdout.readline())
                    assert frame["type"] == name, frame
                    if name == "round_complete":
                        record = frame["record"]
                        assert record["selected_artifact_id"] == (candidate_id if promote else parent_id)
                        assert record["model_call_ids"] == [call_id]
                        assert record["history_ids"] == [item["id"] for item in history]
                        assert record["configuration"]["plan_source"] == ("rules_after_model_reply" if invalid_reply else "pytrio")
                        emit(process, {"type": "round_ack", "accepted": True})
                        break
                    assert frame["id"] == sequence, frame
                    if name == "history_request":
                        reply = {"type": "history_response", "records": history}
                    elif name == "model_request":
                        text = frame["messages"][0]["content"]
                        assert prompt not in text
                        assert all(item["id"] in text for item in history)
                        reply_text = 'experiment(name="local_test", data="full", epochs=2, learning_rate=0.001, batch_size=128, reason="延续当前训练")'
                        if invalid_reply:
                            reply_text = "unparseable test fixture"
                        reply = {"type": "model_response", "call_id": call_id,
                                 "response": {"choices": [{"message": {"role": "assistant", "content": reply_text}}]}}
                    else:
                        assert frame["parent_artifact_id"] == parent_id
                        assert frame["parent_checkpoint_sha256"] == parent_digest
                        assert frame["artifact"]["predict_argv"] == PREDICT_ARGV
                        files = {item["path"]: base64.b64decode(item["data_base64"], validate=True) for item in frame["files"]}
                        audit = json.loads(files["training.json"])
                        assert audit["parent_checkpoint_sha256"] == parent_digest
                        assert audit["checkpoint_sha256"] == digest(files["checkpoint.npz"]) != parent_digest
                        assert audit["final_steps"] > audit["initial_steps"]
                        if task == "B":
                            assert audit["target_unit"] == "minutes"
                        assert "strategy.py" in frame["artifact"]["code_paths"]
                        candidate.mkdir()
                        for filename in ("checkpoint.npz", "training.json", "history.jsonl"):
                            (candidate / filename).write_bytes(files[filename])
                        reply = {"type": "candidate_response", "artifact_id": candidate_id,
                                 "summary": {"metrics": {metric: after}}}
                    emit(process, {"id": sequence, **reply})
                assert process.wait(timeout=15) == 0
                assert not process.stdout.read(), "unexpected frames after round_ack"
            except Exception:
                process.kill()
                process.wait(timeout=15)
                stderr.seek(0)
                print(stderr.read())
                raise
    return candidate, candidate_id, kind


def check_prediction(image, task, public, kind, candidate, candidate_id):
    with (public / "validation.jsonl").open(encoding="utf-8") as f:
        request = {"task": kind, "inputs": json.loads(next(f))["inputs"]}
    context = {"protocol": "contest.predict-context.v1", "dataset_key": task, "task": kind,
               "artifact": {"checkpoint_paths": ["checkpoint.npz"]}, "artifact_id": candidate_id}
    command = docker(image, "predict", mounts=((candidate, "/model"),), interactive=True, environment=(
        ("CONTEST_ARTIFACT_PATH", "/model/checkpoint.npz"), ("CONTEST_PREDICTION_CONTEXT", json.dumps(context)),
    ))
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8") as process:
        emit(process, request)
        output = json.loads(process.stdout.readline())["outputs"]  # The host has not sent EOF.
        process.stdin.close()
        assert process.wait(timeout=15) == 0
    if kind == "classification":
        assert len(output) == 6
    else:
        assert output["value"] >= 0


def project_public(source, target):
    target.mkdir()
    for name in ("train.jsonl", "validation.jsonl"):
        with (source / name).open(encoding="utf-8") as rows, (target / name).open("w", encoding="utf-8") as output:
            for line in rows:
                if line.strip():
                    row = json.loads(line)
                    output.write(json.dumps({key: row[key] for key in ("id", "inputs", "answer")}) + "\n")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="smart-training-harness:competition-v5")
    args = parser.parse_args()
    description = json.loads(subprocess.check_output(docker(args.image, "describe"), encoding="utf-8"))
    assert description["protocol"] == PROTOCOL and description["sdk_version"] == "1.0.2"
    assert description["model_access"] == "host_gateway_rpc" and set(description["tasks"]) == {"classification", "regression"}
    with tempfile.TemporaryDirectory(prefix="contest-sdk-check-", dir=ROOT / ".pytmp") as directory:
        directory = Path(directory)
        for task, pack in PACKS.items():
            official = ROOT / "data" / "official_baselines" / pack
            public = project_public(official / "public", directory / f"public-{task}")
            parent_id = f"official-{task}"
            candidate, identifier, kind = round_trip(args.image, task, directory, public, official,
                                                      parent_id, 1, [], promote=task != "B")
            check_prediction(args.image, task, public, kind, candidate, identifier)
            if task == "A":
                history = [{"id": "history-A-1", "configuration": {"epochs": 2}, "reason": "first round"}]
                round_trip(args.image, task, directory, public, candidate, identifier, 2, history,
                           promote=False)
            print(f"{task}: SDK round, bound checkpoint, decision, and streaming prediction passed", flush=True)


if __name__ == "__main__":
    main()
