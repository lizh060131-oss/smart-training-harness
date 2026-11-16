# -*- coding: utf-8 -*-
"""智训引擎 v2 —— 官方 NumPy-MLP 检查点续训协议的自主漂移适应引擎。

工业模型 = 官方协议检查点（baselines.model.Model，NPZ 无 pickle，<=16MiB）。
每轮流程：读历史 -> 漂移诊断 -> 规划 >=3 个实验（全部从"已采纳检查点"续训）
-> 在当批留存验证切片上评估 -> 门控（部署新检查点 / 保持 / 回滚）-> 完整记录。
第一轮必须从官方基线检查点起步；后续轮从上一轮被采纳的检查点续训。
特征标准化/目标缩放由检查点冻结（官方规则），引擎不重拟合。

LLM 顾问：设置 PETRIO_API_KEY / PETRIO_BASE_URL / PETRIO_MODEL 后由 PyTRIO
千问模型规划实验（OpenAI 兼容接口，stdlib urllib 实现，无第三方依赖）；
未配置时回退到规则规划器并如实记录。
"""
import inspect
import json
import math
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

try:
    import swanlab as _swanlab
except ImportError:  # SwanLab 不可用时静默降级为本地 JSONL 记录
    _swanlab = None

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 多核加速：官方 model.py 在 import 时读取 BASELINE_CPU_THREADS（1-8），
# 必须在此 import 之前设置才生效。评测环境为 4 核，默认用 4。
os.environ.setdefault("BASELINE_CPU_THREADS", "4")

from baselines.model import Model, checkpoint_digest  # noqa: E402
from baselines.data import PreparedRows, json_value, read_dataset, flatten  # noqa: E402
from baselines.train import train_run, metrics         # noqa: E402
from bearing_gate import bearing_replay_groups, bearing_group_rmse, bearing_selection_score  # noqa: E402
from pytrio_tools import request_experiment_plan        # noqa: E402

FORMAT_VERSION = "contest.harness-engine.v2"


# ---------------------------------------------------------------------------
# JSONL 工具
# ---------------------------------------------------------------------------

def read_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json_value(line))
    return rows


def write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_split(rows, val_ratio, seed, classes=None):
    """分类按 answer 分层；回归按 answer 分位数分层。返回 (train, val)。"""
    rng = random.Random(seed)
    if classes:
        by_class = {}
        for r in rows:
            by_class.setdefault(r["answer"], []).append(r)
        train, val = [], []
        for _c, rs in by_class.items():
            rs = list(rs)
            rng.shuffle(rs)
            k = max(1, int(round(len(rs) * val_ratio)))
            val += rs[:k]
            train += rs[k:]
    else:
        rows = list(rows)
        rng.shuffle(rows)
        k = max(1, int(round(len(rows) * val_ratio)))
        val, train = rows[:k], rows[k:]
    return train, val


def dedup_by_id(rows):
    seen, out = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def feature_matrix(rows):
    """rows -> (paths, X)。paths 为官方 flatten 的 JSON 编码路径。"""
    paths = None
    xs = []
    for r in rows:
        p, v = flatten(r["inputs"])
        if paths is None:
            paths = p
        if p != paths:
            raise ValueError("feature schema changed between rows")
        xs.append(v)
    return paths, np.asarray(xs, dtype=np.float64)


# ---------------------------------------------------------------------------
# 漂移监测器（官方特征 schema）
# ---------------------------------------------------------------------------

def compute_psi(expected, actual, bins=10):
    lo, hi = float(expected.min()), float(expected.max())
    if hi - lo < 1e-12:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0].astype(np.float64) + 1e-6
    a = np.histogram(np.clip(actual, lo, hi), bins=edges)[0].astype(np.float64) + 1e-6
    e /= e.sum()
    a /= a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def js_divergence(p, q):
    p = np.asarray(p, dtype=np.float64) + 1e-12
    q = np.asarray(q, dtype=np.float64) + 1e-12
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(0.5 * np.sum(np.where(p > 0, p * np.log(p / m), 0.0)) +
                     0.5 * np.sum(np.where(q > 0, q * np.log(q / m), 0.0)))


class DriftMonitor:
    def __init__(self, task, classes):
        self.task = task
        self.classes = classes or []

    def diagnose(self, reference_rows, new_rows, label):
        _ref_paths, ref_x = feature_matrix(reference_rows)
        new_paths, new_x = feature_matrix(new_rows)
        if _ref_paths != new_paths:
            raise ValueError("feature schema drift between batches")
        d = {"label": label, "n_ref": len(ref_x), "n_new": len(new_x)}
        ref_mean, ref_std = ref_x.mean(axis=0), ref_x.std(axis=0)
        ref_std[ref_std < 1e-12] = 1.0
        new_mean = new_x.mean(axis=0)
        shift = (new_mean - ref_mean) / ref_std
        order = np.argsort(-np.abs(shift))
        d["top_mean_shift"] = [
            {"feature": new_paths[int(i)], "shift_std_units": float(shift[i]),
             "ref_mean": float(ref_mean[i]), "new_mean": float(new_mean[i])}
            for i in order[:8]]
        d["max_abs_shift"] = float(np.max(np.abs(shift)))
        d["mean_abs_shift"] = float(np.mean(np.abs(shift)))
        psis = [compute_psi(ref_x[:, i], new_x[:, i]) for i in range(ref_x.shape[1])]
        order_p = np.argsort(-np.asarray(psis))
        d["top_psi"] = [
            {"feature": new_paths[int(i)], "psi": float(psis[i])} for i in order_p[:5]]
        d["max_psi"] = float(max(psis))
        d["mean_psi"] = float(np.mean(psis))
        if self.task == "classification":
            ref_cnt = np.zeros(len(self.classes), dtype=np.int64)
            new_cnt = np.zeros(len(self.classes), dtype=np.int64)
            for r in reference_rows:
                ref_cnt[self.classes.index(r["answer"])] += 1
            for r in new_rows:
                new_cnt[self.classes.index(r["answer"])] += 1
            d["class_js"] = js_divergence(ref_cnt, new_cnt)
            d["ref_class_dist"] = {c: int(n) for c, n in zip(self.classes, ref_cnt)}
            d["new_class_dist"] = {c: int(n) for c, n in zip(self.classes, new_cnt)}
            d["missing_classes"] = [c for c, n in zip(self.classes, new_cnt) if n == 0]
        d["summary"] = self._summary_text(d)
        return d

    def _summary_text(self, d):
        parts = ["%s: %d samples vs %d historical" % (d["label"], d["n_new"], d["n_ref"])]
        if d.get("max_abs_shift", 0) > 2.0:
            parts.append("SENSOR_RESPONSE_DRIFT (max mean-shift %.2f std)" % d["max_abs_shift"])
        if d.get("max_psi", 0) > 0.25:
            parts.append("DISTRIBUTION_DRIFT (max PSI %.3f)" % d["max_psi"])
        if d.get("class_js", 0) > 0.05:
            parts.append("LABEL_DISTRIBUTION_SHIFT (JS %.3f)" % d["class_js"])
        if d.get("missing_classes"):
            parts.append("CLASS_ABSENCE: " + ",".join(d["missing_classes"]))
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

def eval_checkpoint(ckpt_path, data_path, task, classes, prepared=None, bearing_groups=None):
    model = Model.load(ckpt_path)
    data = prepared if prepared is not None else read_dataset(data_path, task, classes, model.meta["paths"])
    if data["paths"] != model.meta["paths"]:
        raise ValueError("evaluation feature schema does not match the checkpoint")
    if not bearing_groups:
        return metrics(model, data)
    predictions = model.predict(data["x"])
    result = metrics(model, data, predicted=predictions)
    result["bearing_rmse"] = bearing_group_rmse(predictions, data["y"], bearing_groups)
    return result


# ---------------------------------------------------------------------------
# LLM 顾问（PyTRIO / Petrel，OpenAI 兼容）
# ---------------------------------------------------------------------------

def llm_chat(cfg, messages, timeout=90, *, max_tokens=None):
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    payload = {"model": cfg["model"], "messages": messages, "temperature": 0.2}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + cfg["api_key"]})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def llm_available():
    return bool(os.environ.get("PETRIO_API_KEY"))


def get_llm_cfg():
    return {
        "api_key": os.environ.get("PETRIO_API_KEY", ""),
        "base_url": os.environ.get("PETRIO_BASE_URL", ""),
        "model": os.environ.get("PETRIO_MODEL", "qwen3.5-4b"),
    }


# ---------------------------------------------------------------------------
# 实验规划器
# ---------------------------------------------------------------------------

class Planner:
    """保留经过验证的规则候选，再加入不同配置的模型候选。"""

    def __init__(self, task, classes, llm=None, tuning=None, policy_dir=None, audit=None):
        self.task = task
        self.classes = classes or []
        self.tuning = tuning or {}
        self.llm = llm or (get_llm_cfg() if llm_available() else None)
        self.policy_dir = policy_dir
        self.audit = audit

    def _rules(self, diag):
        exps = [
            {"name": "full_hist", "data": "full", "epochs": 3,
             "learning_rate": 0.001, "batch_size": 128,
             "reason": "全历史继续训练，均衡新旧数据"},
            {"name": "recent_double", "data": "recent2", "epochs": 3,
             "learning_rate": 0.001, "batch_size": 128,
             "reason": "额外加入两份当前批样本，加速适应漂移"},
        ]
        if self.task == "classification":
            exps.append({"name": "compact_recent", "data": "compact_recent", "epochs": 3,
                         "learning_rate": 0.001, "batch_size": 128,
                         "reason": "强化当前批，限制历史样本量并保留每个历史类别"})
        else:
            exps.append({"name": "low_lr_long", "data": "full", "epochs": 6,
                         "learning_rate": 0.0005, "batch_size": 256,
                         "reason": "低学习率长训练，平滑适应漂移"})
        # 漂移诊断反馈：剧烈漂移时加大对新数据的强调
        if diag.get("max_abs_shift", 0) > 3.0 or diag.get("max_psi", 0) > 0.5:
            exps.insert(0, {"name": "drift_focus", "data": "recent3", "epochs": 4,
                            "learning_rate": 0.0015, "batch_size": 128,
                            "reason": "诊断到剧烈漂移，额外加入三份当前批样本强化适应"})
        # 每任务可覆盖超参（configs/{TASK}.json 的 tuning 段），键为实验名
        for e in exps:
            ov = self.tuning.get(e["name"], {})
            for k in ("data", "epochs", "learning_rate", "batch_size"):
                if k in ov:
                    e[k] = ov[k]
        return exps[:4]

    def plan(self, round_no, diag, prev_results, min_experiments=3):
        self.llm_error = None
        self.plan_source = "rules"
        self.tool_protocol = None
        rules = self._rules(diag)
        if not self.llm:
            return rules
        try:
            if min_experiments == 3:
                plan = self._hybrid_plan(round_no, diag, prev_results, rules,
                                         2 if self.task == "classification" else 3)
                self.plan_source = "hybrid"
            else:
                plan = self._llm_plan(round_no, diag, prev_results, min_experiments)
                if len(plan) < min_experiments:
                    raise ValueError("LLM returned %d experiments, expected at least %d"
                                     % (len(plan), min_experiments))
                self.plan_source = "pytrio"
            return plan
        except Exception as exc:  # LLM 失败回退规则，如实记录
            self.llm_error = "%s: %s" % (type(exc).__name__, exc)
            return rules

    def _hybrid_plan(self, round_no, diag, prev_results, rules, anchor_count):
        def signature(exp):
            return (exp["data"], exp["epochs"], exp["learning_rate"], exp["batch_size"])

        fixed = []
        fixed_configs = set()
        for exp in rules:
            config = signature(exp)
            if config not in fixed_configs:
                fixed.append(exp)
                fixed_configs.add(config)
            if len(fixed) == anchor_count:
                break
        if len(fixed) != anchor_count:
            raise ValueError("hybrid planning lacks distinct rule configurations")
        for exp in self._llm_plan(round_no, diag, prev_results, min_experiments=2, fixed=fixed):
            if signature(exp) not in fixed_configs:
                if any(exp["name"] == rule["name"] for rule in fixed):
                    exp = {**exp, "name": "pytrio_" + exp["name"][:33]}
                return fixed + [exp]
        raise ValueError("LLM returned no valid configuration distinct from the rule candidates")

    def _llm_plan(self, round_no, diag, prev_results, min_experiments=3, fixed=()):
        history = []
        for r in prev_results[-6:]:
            history.append({"round": r["round"], "decision": r["decision"],
                            "static_primary": r["static_primary"],
                            "adopted_zero_primary": r["adopted_zero_primary"],
                            "adopted_after_primary": r["adopted_after_primary"],
                            "gain_vs_static": r["gain_vs_static"],
                            "chosen_experiment": r["chosen_experiment"],
                            "experiments": r["experiments"],
                            "diagnosis": r["diagnosis"]})
        prompt = (
            "你是工业模型自动训练引擎的实验规划器。任务：%s。\n"
            "读取诊断和历史后提出 %d 个对照候选，考虑当前批改善、历史遗忘与 CPU 预算。"
            "full 使用可用历史，recent1/recent2/recent3 额外加入一/二/三份当前批，"
            "balanced 按类补样，compact_recent 使用限额的分层历史与两份当前批。"
            "必须沿用当前检查点、Adam 状态和原始输入。"
            % (self.task, min_experiments))
        if fixed:
            prompt += ("\n本轮已有以下规则实验。请选择不同于这些实验的训练配置，按优先级排序。\n%s"
                       % json.dumps(fixed, ensure_ascii=False))
        strategies = ("full", "recent1", "recent2", "recent3")
        if self.classes:
            strategies += ("balanced", "compact_recent")
        result = request_experiment_plan(
            lambda messages, **kwargs: llm_chat(self.llm, messages, **kwargs), prompt,
            strategy_source=inspect.getsource(type(self)._rules), history=history, diagnosis=diag,
            policy_path=self.policy_dir / f"round{round_no:02d}.py" if self.policy_dir is not None else None,
            audit=(lambda event: self.audit({**event, "round": round_no})) if self.audit else None,
            data_strategies=strategies,
        )
        self.tool_protocol = result["tool_protocol"]
        exps = []
        for e in result["experiments"]:
            try:
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", e["name"]):
                    raise ValueError("invalid experiment name from LLM")
                e["epochs"] = max(1, min(8, int(e["epochs"])))
                learning_rate = float(e["learning_rate"])
                if not math.isfinite(learning_rate):
                    raise ValueError("invalid learning rate from LLM")
                e["learning_rate"] = max(1e-5, min(3e-3, learning_rate))
                e["batch_size"] = int(e["batch_size"])
                if e["batch_size"] not in (64, 128, 256):
                    raise ValueError("invalid batch size from LLM")
                if not self.classes and e["data"] == "balanced" and self.tool_protocol == "legacy":
                    e["data"] = "full"
                if e["data"] not in strategies:
                    raise ValueError("bad data strategy from LLM: %r" % e["data"])
            except (KeyError, TypeError, ValueError, OverflowError):
                if fixed:
                    continue
                raise
            exps.append(e)
        return exps[:max(4, min_experiments)]


# ---------------------------------------------------------------------------
# 门控器
# ---------------------------------------------------------------------------

class Gate:
    """在当批留存集上择优部署，用累积 replay 池检查历史性能。

    replay 防遗忘预算以**静态基线在 replay 上的表现**为固定参照（累计预算），
    而不是与上一轮已采纳模型逐轮比较，避免"每轮小退化叠加"导致永远无法部署。
    """

    def __init__(self, task, deploy_margin=None, soft_budget=None, hard_budget=None):
        self.task = task
        if deploy_margin is None:
            deploy_margin = 0.003 if task == "classification" else 0.002  # 相对
        if soft_budget is None:
            soft_budget = 0.08 if task == "classification" else 0.10   # 相对
        if hard_budget is None:
            hard_budget = 0.30 if task == "classification" else 0.40   # 相对
        self.deploy_margin = deploy_margin
        self.soft_budget = soft_budget
        self.hard_budget = hard_budget

    def primary(self, m):
        return m["macro_f1"] if self.task == "classification" else m["rmse"]

    def better(self, a, b):
        """a 是否显著优于 b（a/b 为 primary 值）。"""
        if self.task == "classification":
            return a >= b + self.deploy_margin
        return a <= b * (1.0 - self.deploy_margin)

    def replay_loss(self, replay_val, baseline_replay):
        if self.task == "classification":
            return baseline_replay - replay_val
        return (replay_val - baseline_replay) / max(abs(baseline_replay), 1e-9)

    def needs_rollback(self, replay_val, baseline_replay):
        return (replay_val is not None and baseline_replay is not None
                and self.replay_loss(replay_val, baseline_replay) > self.hard_budget)

    def replay_verdict(self, replay_val, baseline_replay, adopted_replay, gain):
        """两级 replay 防遗忘判定，返回 (ok, reason)。

        gain：候选相对已采纳模型在当批上的改进量（分类=macro_f1 差，回归=相对
        下降比例）。loss≤soft_budget → 直接通过；loss>hard_budget → 灾难遗忘拒绝
        （对基线的绝对下限）；介于之间 → 边际豁免：候选 replay 不劣于当前已采纳
        （marginal≥0，严格占优），或当批改进 ≥ 边际 replay 损失才通过。
        """
        loss = self.replay_loss(replay_val, baseline_replay)
        if loss <= self.soft_budget:
            return True, "replay stable (loss=%.4f)" % loss
        if loss > self.hard_budget:
            return False, "catastrophic forgetting (loss=%.4f > %.2f)" % (
                loss, self.hard_budget)
        marginal = -self.replay_loss(replay_val, adopted_replay)
        if marginal >= 0:
            return True, "replay improves vs adopted (marginal=+%.4f)" % marginal
        if gain >= -marginal:
            return True, "net-win marginal waiver (gain=%.4f >= replay loss=%.4f)" % (
                gain, -marginal)
        return False, "replay degradation uncompensated (gain=%.4f < loss=%.4f)" % (
            gain, -marginal)

    def decide(self, candidates, adopted, parent, static_primary, baseline_replay, baseline_bearings=None):
        """candidates=[{name,primary_val,replay_val,ckpt,...}]（primary_val 已算好）。

        规则：先过滤出「当批显著优于已采纳」且「replay 通过两级防遗忘」的候选，
        在其中选当批最优 → DEPLOY_NEW；无候选时若已采纳模型 replay 灾难遗忘，
        且父检查点在相同验证集上未明显退化、replay 恢复到硬阈值以内时，回滚到父检查点。
        """
        base = adopted["primary_val"] or 0.0
        eligible = []
        for c in candidates:
            if not self.better(c["primary_val"], adopted["primary_val"]):
                continue
            if self.task == "classification":
                gain = c["primary_val"] - adopted["primary_val"]
            else:
                gain = (adopted["primary_val"] - c["primary_val"]) / max(abs(base), 1e-9)
            if baseline_replay is None or c["replay_val"] is None \
                    or adopted["replay_val"] is None:
                ok, why = True, "no replay pool yet (round 1)"
            else:
                ok, why = self.replay_verdict(c["replay_val"], baseline_replay,
                                              adopted["replay_val"], gain)
            if ok:
                eligible.append((c, why))
        best = None
        if eligible:
            best = (max(eligible, key=lambda t: t[0]["primary_val"])[0]
                    if self.task == "classification"
                    else min(eligible, key=lambda t: t[0]["primary_val"])[0])
        reasons = ["%d/%d candidates pass primary+replay gate (adopted=%.4f, baseline_replay=%s)"
                   % (len(eligible), len(candidates), adopted["primary_val"],
                      "None" if baseline_replay is None else "%.4f" % baseline_replay)]
        if best is not None and baseline_bearings:
            best, score = max(((c, bearing_selection_score(c, adopted, baseline_bearings))
                               for c, _ in eligible), key=lambda item: item[1])
            reasons.append("best bearing-balanced gain=%.6f across %d historical bearings"
                           % (score, len(baseline_bearings)))
            if score <= 0:
                best = None
        if best is not None:
            reasons.append("deploy %s primary=%.4f replay=%.4f" %
                           (best["name"], best["primary_val"], best["replay_val"]))
            return {"action": "DEPLOY_NEW", "chosen": best, "reasons": reasons}
        # 回滚检查：已采纳模型历史遗忘超硬预算（灾难性），且存在父检查点
        if parent is not None and self.needs_rollback(adopted["replay_val"], baseline_replay):
            parent_loss = self.replay_loss(parent["replay_val"], baseline_replay)
            reasons.append("parent %s reevaluated: primary=%.4f replay=%.4f loss=%.4f"
                           % (parent["name"], parent["primary_val"], parent["replay_val"], parent_loss))
            parent_primary = parent["primary_val"]
            if (parent_loss <= self.hard_budget and (parent_primary == adopted["primary_val"]
                    or not self.better(adopted["primary_val"], parent_primary))):
                reasons.append("adopted replay loss %.4f beyond hard budget; ROLLBACK to parent %s"
                               % (self.replay_loss(adopted["replay_val"], baseline_replay), parent["name"]))
                return {"action": "ROLLBACK", "chosen": None, "reasons": reasons}
            reasons.append("parent does not pass the current rollback checks")
        reasons.append("no candidate passes gate")
        return {"action": "KEEP", "chosen": None, "reasons": reasons}


# ---------------------------------------------------------------------------
# 记录器（SwanLab 兼容：JSONL 事件 + 汇总 JSON）
# ---------------------------------------------------------------------------

class Recorder:
    def __init__(self, out_dir, title):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "engine_log.jsonl"
        self.title = title
        # 每次运行从新日志开始，避免跨 run 累积
        with open(self.log_path, "w", encoding="utf-8", newline="\n"):
            pass

    def emit(self, event):
        event["ts"] = time.time()
        with open(self.log_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def write_summary(self, summary):
        with open(self.out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# SwanLab 记录（工业模型续训过程；规则要求公开可见）
# ---------------------------------------------------------------------------

class SwanLabMirror:
    """把引擎事件镜像到 SwanLab（记录对象=被训练的工业模型）。

    mode 由环境变量决定：SWANLAB_MODE=cloud|local|disabled（默认有
    SWANLAB_API_KEY 时 cloud，否则 local）。SwanLab 不可用或失败时静默
    降级，engine_log.jsonl 始终是权威记录。
    """

    def __init__(self, task, out_dir):
        self.run = None
        self.error = None
        self.mode = None
        if _swanlab is None:
            self.error = "swanlab not importable"
            return
        mode = os.environ.get("SWANLAB_MODE") or "cloud"
        if mode == "disabled":
            return
        project = os.environ.get("SWANLAB_PROJECT_NAME", "smart-training-harness")

        def _init(m):
            return _swanlab.init(
                workspace=os.environ.get("SWANLAB_WORKSPACE"),
                project=project,
                experiment_name="%s-%s" % (task, time.strftime("%Y%m%d-%H%M%S")),
                mode=m,
                logdir=str(Path(out_dir) / "swanlog"),
                description="智训引擎 v2 工业模型自主续训（任务 %s）" % task)

        try:
            self.run = _init(mode)
            self.mode = mode
        except Exception as exc1:
            if mode == "cloud":  # 云端不可用（未登录/无网络）→ 本地兜底
                try:
                    self.run = _init("local")
                    self.mode = "local(fallback)"
                except Exception as exc2:
                    self.error = "cloud: %s: %s | local: %s: %s" % (
                        type(exc1).__name__, exc1, type(exc2).__name__, exc2)
                    self.run = None
            else:
                self.error = "%s: %s" % (type(exc1).__name__, exc1)
                self.run = None

    def log(self, data, step=None):
        if self.run is None:
            return
        try:
            self.run.log(data, step=step)
        except Exception:
            pass

    def finish(self):
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 轮次引擎
# ---------------------------------------------------------------------------

class RoundEngine:
    def __init__(self, task, profile_path, baseline_ckpt, batch_files, out_dir,
                 rounds=None, val_ratio=0.2, replay_ratio=0.1, seed=42,
                 llm=None, min_experiments=3):
        self.task = task
        self.profile_path = str(profile_path)
        self.profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        self.classes = self.profile.get("classes") or []
        self.baseline_ckpt = str(baseline_ckpt)
        self.batch_files = [str(p) for p in batch_files]
        self.rounds = rounds or len(self.batch_files)
        self.val_ratio = val_ratio
        self.replay_ratio = replay_ratio
        self.seed = seed
        self.min_experiments = min_experiments
        self.task_type = self.profile["task"]  # classification / regression
        self.out_dir = Path(out_dir)
        self.monitor = DriftMonitor(self.task_type, self.classes)
        self.recorder = Recorder(self.out_dir, "harness-v2 task=%s" % task)
        self.planner = Planner(self.task_type, self.classes, llm=llm,
                               tuning=self.profile.get("tuning"), policy_dir=self.out_dir / "policies",
                               audit=self.recorder.emit)
        # 训练历史窗口（批数）：只保留最近 N 批历史续训（None=全历史）。
        # 对有真实漂移的任务可丢弃过旧数据并显著提速；无漂移任务保持全历史。
        self.history_window = self.profile.get("history_window")
        self.gate = Gate(self.task_type)
        self.swanlab = SwanLabMirror(task, out_dir)
        self.rng = random.Random(seed)

    # ---- 数据准备 ----

    def prepare_replay(self, first_batch_rows):
        replay, _keep = stratified_split(first_batch_rows, self.replay_ratio,
                                         self.seed, self.classes)
        replay_path = self.out_dir / "replay.jsonl"
        write_rows(replay_path, replay)
        return replay_path, replay

    def build_train_variant(self, cum_train, current_train, strategy, suffix):
        """data 策略 -> 训练 JSONL（重复行必须换新 id 以保持 id 唯一）。"""
        if strategy == "full":
            rows = self._history_rows(cum_train)
        elif strategy == "balanced":
            rows = self._balanced_rows(self._history_rows(cum_train), suffix)
        elif strategy in ("recent1", "recent2", "recent3"):
            rows = self._history_rows(cum_train) + self._dup(current_train, int(strategy[-1]), suffix)
        elif strategy == "compact_recent" and self.task_type == "classification":
            current_ids = {row["id"] for row in current_train}
            history_by_class = {}
            for row in self._history_rows(cum_train):
                if row["id"] not in current_ids:
                    history_by_class.setdefault(row["answer"], []).append(row)
            budget = max(512, min(2400, len(current_train)))
            quota = budget // max(1, len(history_by_class))
            sampler = random.Random(self.seed + len(self._train_stack))
            history = [row for group in history_by_class.values()
                       for row in (group if len(group) <= quota else sampler.sample(group, quota))]
            rows = history + current_train + self._dup(current_train, 1, suffix)
        else:
            raise ValueError("unknown data strategy %r" % strategy)
        return rows

    def _history_rows(self, cum_train):
        """续训历史：history_window 设了且已有更多批时，只取最近 N 批。"""
        if self.history_window and len(self._train_stack) > self.history_window:
            rows = []
            for batch in self._train_stack[-self.history_window:]:
                rows += list(batch)
            return rows
        return list(cum_train)

    def _dup(self, rows, times, suffix):
        out = []
        for t in range(times):
            for i, r in enumerate(rows):
                copy = dict(r)
                copy["id"] = "%s_%s%d" % (r["id"], suffix, t)
                out.append(copy)
        return out

    def _balanced_rows(self, rows, suffix):
        if self.task != "classification":
            return list(rows)
        by_class = {}
        for r in rows:
            by_class.setdefault(r["answer"], []).append(r)
        target = max(len(v) for v in by_class.values())
        out = []
        for c, rs in by_class.items():
            out += rs
            extra = target - len(rs)
            if extra > 0:
                out += self._dup(rs[:extra], 1, "%s_bal_%s" % (suffix, c))
        return out

    # ---- 主循环 ----

    def run(self):
        rec = self.recorder
        rec.emit({"type": "run_start", "format": FORMAT_VERSION, "task": self.task,
                  "profile": self.profile_path, "baseline": checkpoint_digest(self.baseline_ckpt),
                  "rounds": self.rounds, "batches": self.batch_files,
                  "llm": {"enabled": bool(self.planner.llm),
                          "model": self.planner.llm["model"] if self.planner.llm else None},
                  "swanlab": {"error": self.swanlab.error}})
        self.swanlab.log({"config/rounds": self.rounds,
                          "config/val_ratio": self.val_ratio,
                          "config/replay_ratio": self.replay_ratio,
                          "config/seed": self.seed}, step=0)

        # 跨历史批次的累积 replay 池（每批切一小片永入池、不进训练，防遗忘指示器；
        # 至少积累一个批次后（第 2 轮起）才启用 replay 防遗忘检查）
        replay_pool = []
        replay_path = self.out_dir / "replay.jsonl"

        adopted_ckpt = self.baseline_ckpt
        adopted_meta = {"name": "official_baseline", "ckpt": adopted_ckpt}
        adopted_chain = [adopted_meta]           # 采纳链，chain[-2] 为父检查点
        cum_train, released_all = [], []          # released_all 含历史 train+val
        self._train_stack = []                    # 每批 train_new，供 history_window 使用
        rounds_summary = []
        datasets = PreparedRows(self.task_type, self.classes)

        for b in range(1, self.rounds + 1):
            incoming = read_rows(self.batch_files[b - 1])
            train_new, val_new = stratified_split(incoming, self.val_ratio,
                                                  self.seed + b, self.classes)
            # 从本批训练部分再切一小片进 replay 池（永不训练）
            train_new, replay_slice = stratified_split(train_new, self.replay_ratio,
                                                       self.seed + 1000 + b, self.classes)
            prior_ids = {row["id"] for row in replay_pool} if self.task == "B" else set()
            if replay_slice:
                replay_pool = dedup_by_id(replay_pool + replay_slice)
            bearing_groups = bearing_replay_groups(replay_pool, prior_ids) if prior_ids else {}
            val_path = self.out_dir / ("round%02d_val.jsonl" % b)
            val_data = datasets.write(val_path, val_new)

            # 本轮 replay 池（第 2 轮起启用）
            baseline_replay = None
            baseline_bearings = {}
            replay_data = None
            if replay_pool:
                replay_data = datasets.write(replay_path, replay_pool)
                baseline_scores = eval_checkpoint(self.baseline_ckpt, str(replay_path),
                                                  self.task_type, self.classes, replay_data, bearing_groups)
                baseline_replay = self.gate.primary(baseline_scores)
                baseline_bearings = baseline_scores.get("bearing_rmse", {})
                rec.emit({"type": "baseline_replay", "round": b,
                          "n_pool": len(replay_pool), "primary": baseline_replay,
                          "bearing_rmse": baseline_bearings})

            # 漂移诊断：新到达批次 vs 全部历史（首轮无历史时与批内训练切片比）
            reference = released_all if released_all else train_new
            diag = self.monitor.diagnose(
                reference, incoming, label="batch %d incoming" % b)
            cum_train = dedup_by_id(cum_train + train_new)
            self._train_stack.append(train_new)
            released_all = dedup_by_id(released_all + incoming)
            rec.emit({"type": "diagnosis", "round": b, "diagnosis": diag})

            # 静态基线在当批留存上的表现（冻结官方检查点）
            static = eval_checkpoint(self.baseline_ckpt, str(val_path), self.task_type, self.classes, val_data)
            static_primary = self.gate.primary(static)
            # 已采纳模型在当批留存上的零样本表现
            adopted_primary = static_primary
            adopted_replay = baseline_replay
            adopted_bearings = baseline_bearings
            if adopted_ckpt != self.baseline_ckpt:
                adopted_primary = self.gate.primary(eval_checkpoint(
                    adopted_ckpt, str(val_path), self.task_type, self.classes, val_data))
                if replay_data is not None:
                    adopted_scores = eval_checkpoint(adopted_ckpt, str(replay_path), self.task_type,
                                                     self.classes, replay_data, bearing_groups)
                    adopted_replay = self.gate.primary(adopted_scores)
                    adopted_bearings = adopted_scores.get("bearing_rmse", {})

            plan = self.planner.plan(b, diag, rounds_summary, self.min_experiments)
            if len(plan) < self.min_experiments:
                raise RuntimeError("planner returned fewer than %d experiments" %
                                   self.min_experiments)
            plan_source = self.planner.plan_source
            tool_protocol = self.planner.tool_protocol
            rec.emit({"type": "plan", "round": b, "source": plan_source,
                      "experiments": plan, "llm_error": self.planner.llm_error, "tool_protocol": tool_protocol})

            candidates = []
            for i, spec in enumerate(plan):
                name = "round%02d-exp%d-%s" % (b, i + 1, spec["name"])
                train_rows = self.build_train_variant(cum_train, train_new,
                                                      spec["data"], name)
                train_path = self.out_dir / (name + "_train.jsonl")
                out_dir_exp = self.out_dir / name
                try:
                    train_data = datasets.write(train_path, train_rows)
                    t0 = time.monotonic()
                    report = train_run(
                        self.profile_path, str(train_path), str(val_path),
                        str(out_dir_exp), resume=adopted_ckpt,
                        epochs=spec["epochs"], learning_rate=spec["learning_rate"],
                        batch_size=spec["batch_size"], stage_id=name, prepared_validation=val_data,
                        prepared_train=train_data)
                except Exception as exc:
                    rec.emit({"type": "experiment", "round": b, "name": name,
                              "status": "failed", "error": "%s: %s" % (type(exc).__name__, exc)})
                    continue
                elapsed = time.monotonic() - t0
                ckpt = str(out_dir_exp / "checkpoint.npz")
                primary_val = self.gate.primary(report["validation_after"])
                replay_scores = (eval_checkpoint(ckpt, str(replay_path), self.task_type, self.classes,
                                                 replay_data, bearing_groups) if replay_data is not None else {})
                replay_val = self.gate.primary(replay_scores) if replay_scores else None
                candidates.append({
                    "name": spec["name"], "stage_id": name, "ckpt": ckpt,
                    "primary_val": primary_val, "replay_val": replay_val,
                    "bearing_rmse": replay_scores.get("bearing_rmse", {}),
                    "val_before": report["validation_before"],
                    "val_after": report["validation_after"],
                    "epochs": spec["epochs"], "learning_rate": spec["learning_rate"],
                    "data": spec["data"], "seconds": elapsed,
                    "parent_digest": report["parent_checkpoint_sha256"],
                    "checkpoint_digest": report["checkpoint_sha256"]})
                rec.emit({"type": "experiment", "round": b, "status": "completed",
                          "candidate": {k: v for k, v in candidates[-1].items()
                                        if k not in ("val_before", "val_after")}})
                self.swanlab.log({
                    "exp/%s/primary" % spec["name"]: primary_val,
                    "exp/%s/replay" % spec["name"]: replay_val if replay_val is not None else float("nan"),
                    "exp/%s/seconds" % spec["name"]: elapsed}, step=b)
                if self.swanlab.run is not None:
                    chart = f"train_round_{b:02d}/experiment_{len(candidates)}"
                    with (out_dir_exp / "history.jsonl").open(encoding="utf-8") as history:
                        for epoch, line in enumerate(history, 1):
                            record = json.loads(line)
                            self.swanlab.log({f"{chart}_loss": record["loss"],
                                              f"{chart}_val": self.gate.primary(record["validation"])}, step=epoch)

            adopted_state = {"primary_val": adopted_primary,
                             "replay_val": adopted_replay,
                             "bearing_rmse": adopted_bearings,
                             "name": adopted_meta["name"]}
            decision = self.gate.decide(candidates, adopted_state, None,
                                        static_primary, baseline_replay, baseline_bearings)
            if (decision["action"] == "KEEP" and len(adopted_chain) >= 2
                    and self.gate.needs_rollback(adopted_state["replay_val"], baseline_replay)):
                p = adopted_chain[-2]
                if p["ckpt"] == self.baseline_ckpt:
                    parent_primary, parent_replay = static_primary, baseline_replay
                else:
                    parent_primary = self.gate.primary(
                        eval_checkpoint(p["ckpt"], str(val_path), self.task_type, self.classes, val_data))
                    parent_replay = self.gate.primary(
                        eval_checkpoint(p["ckpt"], str(replay_path), self.task_type, self.classes, replay_data))
                parent_state = {"name": p["name"], "primary_val": parent_primary,
                                "replay_val": parent_replay}
                decision = self.gate.decide(candidates, adopted_state, parent_state,
                                            static_primary, baseline_replay, baseline_bearings)

            adopted_after = adopted_primary
            if decision["action"] == "DEPLOY_NEW":
                chosen = decision["chosen"]
                adopted_ckpt = chosen["ckpt"]
                adopted_after = chosen["primary_val"]
                adopted_meta = {"name": chosen["stage_id"], "ckpt": adopted_ckpt}
                adopted_chain.append(adopted_meta)
            elif decision["action"] == "ROLLBACK":
                adopted_chain.pop()
                adopted_meta = adopted_chain[-1]
                adopted_ckpt = adopted_meta["ckpt"]
                adopted_after = parent_primary
            # KEEP：沿用当前

            row = {
                "round": b,
                "batch_file": self.batch_files[b - 1],
                "n_incoming": len(incoming), "n_train_cum": len(cum_train),
                "n_val": len(val_new),
                "diagnosis": diag["summary"],
                "plan_source": plan_source,
                "tool_protocol": tool_protocol,
                "static_primary": static_primary,
                "adopted_zero_primary": adopted_primary,
                "adopted_after_primary": adopted_after,
                "decision": decision["action"],
                "chosen_experiment": decision["chosen"]["name"] if decision["chosen"] else None,
                "adopted_stage": adopted_meta["name"],
                "gain_vs_static": adopted_after - static_primary,
                "experiments": [{"name": c["name"], "primary_val": c["primary_val"],
                                 "replay_val": c["replay_val"], "seconds": c["seconds"],
                                 "data": c["data"], "epochs": c["epochs"],
                                 "learning_rate": c["learning_rate"]}
                                for c in candidates],
                "reasons": decision["reasons"],
                "n_experiments": len(candidates),
            }
            rounds_summary.append(row)
            rec.emit({"type": "decision", "round": b, **row})
            self.swanlab.log({
                "round/static_primary": static_primary,
                "round/adopted_zero": adopted_primary,
                "round/adopted_after": row["adopted_after_primary"],
                "round/gain_vs_static": row["gain_vs_static"],
                "round/decision": {"DEPLOY_NEW": 1, "KEEP": 0, "ROLLBACK": -1}.get(
                    decision["action"], 0)}, step=b)
            # 被采纳实验的逐 epoch 训练曲线镜像到 SwanLab（工业模型训练过程）
            if decision["chosen"]:
                hist_path = self.out_dir / decision["chosen"]["stage_id"] / "history.jsonl"
                if hist_path.is_file():
                    for line in Path(hist_path).read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        h = json.loads(line)
                        self.swanlab.log({"train/loss": h.get("loss"),
                                          "train/val_primary": self.gate.primary(
                                              h.get("validation", {})) if h.get("validation") else float("nan")},
                                         step=int(h["epoch"]))

        adopted_final = eval_checkpoint(adopted_ckpt, str(replay_path), self.task_type, self.classes, replay_data)
        static_final = eval_checkpoint(self.baseline_ckpt, str(replay_path), self.task_type, self.classes, replay_data)
        llm_rounds = sum(r["plan_source"] in ("hybrid", "pytrio") for r in rounds_summary)
        llm_tool_rounds = sum(r["plan_source"] in ("hybrid", "pytrio") and r["tool_protocol"] == "policy_dsl"
                              for r in rounds_summary)
        summary = {
            "format": FORMAT_VERSION, "task": self.task, "rounds": rounds_summary,
            "adopted_final_stage": adopted_meta["name"],
            "adopted_final_ckpt": adopted_ckpt,
            "adopted_final_digest": checkpoint_digest(adopted_ckpt),
            "llm_used": llm_rounds > 0,
            "llm_rounds": llm_rounds,
            "llm_tool_rounds": llm_tool_rounds,
            "experiment_count": sum(r["n_experiments"] for r in rounds_summary),
            "decision_count": len(rounds_summary),
            "final_replay": {"adopted": adopted_final, "static": static_final},
        }
        rec.write_summary(summary)
        rec.emit({"type": "run_end", "summary": summary})
        self.swanlab.finish()
        return summary
