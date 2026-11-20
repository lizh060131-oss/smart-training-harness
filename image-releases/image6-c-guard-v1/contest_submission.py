"""SDK entry points for the official CPU trainer included in image 6."""
import json
import math
import sys
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from baselines.model import Model, checkpoint_digest
from baselines.predict import predict as baseline_predict
from baselines.train import train_run
from c_guard import review


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
    parent = rnd.parent.path(declared[0] if len(declared) == 1 else "checkpoint.npz")
    parent_digest = checkpoint_digest(parent)
    metric = "macro_f1" if defaults["task"] == "classification" else "rmse"
    higher = metric == "macro_f1"
    history = rnd.history()
    short = (0.00025, 1, 256)
    regular = (0.001, 3, 128)
    alternatives = {"conservative": (0.0005, 2, 256), "standard": (0.0015, 3, 128)}
    if key == "C":
        short = (0.000025, 1, 256)
        regular = (0.00025, 1, 256)
        alternatives = {"conservative": (0.00005, 1, 256), "standard": (0.0001, 1, 256)}
    prompt = {
        "dataset": key, "round": rnd.round_index, "metric": metric,
        "recent_rounds": [str(row.get("reason", ""))[:300] for row in history[-3:]],
        "fixed_trials_lr_epochs_batch": [short, regular],
        "third_trial_options": alternatives,
        "request": "Choose a complementary third trial. Reply only conservative or standard.",
    }
    receipt = rnd.model([{"role": "user", "content": json.dumps(prompt)}],
                        max_tokens=min(128, rnd.max_output_tokens), temperature=0)
    reply = receipt.text.strip().strip('"').lower()
    option = reply if reply in alternatives else "conservative"
    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    work.mkdir(exist_ok=True, parents=True)
    root = Path(tempfile.mkdtemp(prefix="trials-", dir=work))
    trials = []
    for index, (lr, epochs, batch) in enumerate((short, regular, alternatives[option])):
        result = root / str(index)
        report = train_run(profile, rnd.data.train_path, rnd.data.validation_path, result,
                           resume=parent, epochs=epochs, learning_rate=lr, batch_size=batch,
                           stage_id=rnd.stage_id or str(rnd.round_index))
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
    c_audit, c_state, c_adopt = None, None, None
    if key == "C":
        best, c_adopt, c_audit, c_state = review(rnd, root, parent, trials, best)
    trial = trials[best]
    score = trial["validation_after"][metric]
    previous = trial["validation_before"][metric]
    adopt = score >= previous if higher else score <= previous
    if c_adopt is not None:
        adopt = c_adopt
    config = {"image_release": os.getenv("HARNESS_RELEASE"), "dataset": key, "round": rnd.round_index, "trials": trials, "candidate_trial": best, "third_trial": option,
              "model_choice_recognized": reply in alternatives, "model_call_id": receipt.call_id}
    if c_audit is not None:
        config["c_guard"] = c_audit
    declaration = dict(hypothesis="Compare short and longer updates from the same mounted parent.",
                       change_summary="Ran three CPU baseline updates on the current training projection.",
                       configuration=config, seed=trial["seed"])
    if key == "C":
        declaration.update(hypothesis="Require clean per-subset RMSE and MAE to preserve parent and initial reference performance.",
                           change_summary="Three genuine one-epoch updates on the full current training projection; retain parent unless all checked groups pass and pooled RMSE improves at least 0.2%.")
    submitted = rnd.submit_candidate(root / str(best), checkpoint_paths=["checkpoint.npz"],
                                     code_paths=[], predict_argv=PREDICT_ARGV, **declaration)
    child_id = submitted["artifact_id"]
    selected = child_id if adopt else rnd.parent.artifact_id
    decision = "adopt_child" if adopt else "keep_existing_parent"
    config["decision"] = decision
    if c_state is not None:
        rnd.save_state(c_state)
    rnd.finish(selected_artifact_id=selected, reason=(
        f"{decision}: public {metric}, parent={previous}, best child={score}."),
        **{**declaration, "configuration": {**config, "candidate_artifact_id": child_id,
                                           "parent_artifact_id": rnd.parent.artifact_id,
                                           "decision": decision}})
    print("HARNESS_ROUND " + json.dumps({"configuration": config}), file=sys.stderr, flush=True)


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
