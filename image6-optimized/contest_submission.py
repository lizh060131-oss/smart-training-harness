"""SDK entry points for the official CPU trainer included in image 6."""
import json
import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from baselines.data import read_dataset
from baselines.model import Model, checkpoint_digest
from baselines.predict import predict as baseline_predict
from baselines.train import train_run


PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
CONFIGS = Path(__file__).resolve().parent / "configs"


def train(rnd):
    key = rnd.dataset_key
    if key not in ("A", "B", "C"):
        raise ValueError("unknown dataset")
    profile = CONFIGS / (key + ".json")
    defaults = json.loads(profile.read_bytes())
    if defaults["task"] != rnd.task["task"]:
        raise ValueError("platform task differs from baseline profile")
    declared = rnd.parent.descriptor["checkpoint_paths"]
    checkpoint = declared[0] if len(declared) == 1 else "checkpoint.npz"
    parent = rnd.parent.path(checkpoint)
    parent_digest = checkpoint_digest(parent)
    metric = "macro_f1" if defaults["task"] == "classification" else "rmse"
    higher = metric == "macro_f1"
    history = rnd.history()
    short = (0.00025, 1, 256)
    if key == "B":
        fixed_trials = (short,)
        alternatives = {"conservative": (0.000125, 1, 256), "standard": (0.0005, 1, 128)}
        option_key = "second_trial"
        request = "Choose a complementary second trial. Reply only conservative or standard."
        hypothesis = "Compare two one-epoch updates from the same mounted parent."
        change_summary = "Ran two one-epoch CPU updates on the current training projection."
    else:
        fixed_trials = (short, (0.001, 3, 128))
        alternatives = {"conservative": (0.0005, 2, 256), "standard": (0.0015, 3, 128)}
        option_key = "third_trial"
        request = "Choose a complementary third trial. Reply only conservative or standard."
        hypothesis = "Compare short and longer updates from the same mounted parent."
        change_summary = "Ran three CPU baseline updates on the current training projection."
    prompt = {
        "dataset": key, "round": rnd.round_index, "metric": metric,
        "recent_rounds": [str(row.get("reason", ""))[:300] for row in history[-3:]],
        "fixed_trials_lr_epochs_batch": fixed_trials,
        f"{option_key}_options": alternatives,
        "request": request,
    }
    receipt = rnd.model([{"role": "user", "content": json.dumps(prompt)}],
                        max_tokens=min(128, rnd.max_output_tokens), temperature=0)
    reply = receipt.text.strip().strip('"').lower()
    option = reply if reply in alternatives else "conservative"
    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    work.mkdir(exist_ok=True, parents=True)
    root = Path(tempfile.mkdtemp(prefix="trials-", dir=work))
    classes = defaults.get("classes", [])
    training = read_dataset(rnd.data.train_path, defaults["task"], classes)
    validation = read_dataset(rnd.data.validation_path, defaults["task"], classes, training["paths"])
    trials = []
    for index, (lr, epochs, batch) in enumerate((*fixed_trials, alternatives[option])):
        result = root / str(index)
        report = train_run(profile, rnd.data.train_path, rnd.data.validation_path, result,
                           resume=parent, epochs=epochs, learning_rate=lr, batch_size=batch,
                           stage_id=rnd.stage_id or str(rnd.round_index),
                           prepared_train=training, prepared_validation=validation)
        if (report["parent_checkpoint_sha256"] != parent_digest
                or report["final_steps"] <= report["initial_steps"]
                or report["checkpoint_sha256"] == parent_digest):
            raise ValueError("trainer did not produce a child of the mounted checkpoint")
        trials.append({field: report[field] for field in (
            "learning_rate", "batch_size", "initial_steps", "final_steps", "final_epochs",
            "initial_epochs", "parent_checkpoint_sha256", "checkpoint_sha256",
            "train_rows", "validation_rows", "train_sha256", "validation_sha256",
            "validation_before", "validation_after", "seed")})
    choose = max if higher else min
    best = choose(range(len(trials)), key=lambda i: trials[i]["validation_after"][metric])
    trial = trials[best]
    score = trial["validation_after"][metric]
    previous = trial["validation_before"][metric]
    adopt = score >= previous if higher else score <= previous
    config = {"trials": trials, "candidate_trial": best, option_key: option,
              "model_choice_recognized": reply in alternatives, "model_call_id": receipt.call_id}
    declaration = dict(hypothesis=hypothesis, change_summary=change_summary, configuration=config,
                       seed=trial["seed"])
    submitted = rnd.submit_candidate(root / str(best), checkpoint_paths=["checkpoint.npz"],
                                     code_paths=[], predict_argv=PREDICT_ARGV, **declaration)
    child_id = submitted["artifact_id"]
    selected = child_id if adopt else rnd.parent.artifact_id
    decision = "adopt_child" if adopt else "keep_existing_parent"
    rnd.finish(selected_artifact_id=selected, reason=(
        f"{decision}: public {metric}, parent={previous}, best child={score}."),
        **{**declaration, "configuration": {**config, "candidate_artifact_id": child_id,
                                           "parent_artifact_id": rnd.parent.artifact_id,
                                           "decision": decision}})


@lru_cache(maxsize=1)
def load_model(model_dir, checkpoint):
    root = Path(model_dir).resolve()
    path = (root / checkpoint).resolve()
    if not path.is_relative_to(root):
        raise ValueError("selected checkpoint must be inside the mounted model directory")
    return Model.load(path)


def predict(sample, model_dir):
    model = load_model(model_dir, os.getenv("CONTEST_ARTIFACT_PATH", "checkpoint.npz"))
    if model.meta["task"] != sample.task or (
            sample.dataset_key and model.meta["profile"] != sample.dataset_key):
        raise ValueError("prediction context differs from selected checkpoint")
    outputs = baseline_predict(model, sample.inputs)["outputs"]
    if sample.task == "classification":
        winner = max(outputs, key=lambda label: label["score"])
        best_score = winner["score"]
        for label in outputs:
            if label is not winner and label["score"] == best_score:
                label["score"] = math.nextafter(best_score, 0.0)
    return outputs
