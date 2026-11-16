# -*- coding: utf-8 -*-
"""在 PyTRiO chat 接口上运行两步的受限训练策略工具。"""
import ast
import hashlib
import json
import re
import tempfile
from pathlib import Path


DEFAULT_STRATEGIES = ("full", "balanced", "recent2", "recent3")
RESPONSE_LIMIT = 16_000
POLICY_LIMIT = 6_000
FIELDS = {"name", "data", "epochs", "learning_rate", "batch_size", "reason"}
FIRST_INSTRUCTION = (
    "\n使用训练工具协议。第一步只返回 JSON 对象 "
    '{"tool":"read_context","args":{}}，读取可用规则源码、实际诊断与历史实验。'
    "工具返回上下文后才编写策略。"
)
WRITE_INSTRUCTION = (
    "\n现在调用 write_strategy。只返回 JSON 对象 "
    '{"tool":"write_strategy","args":{"code":"<策略代码>"}}。'
    "code 中每个候选是独立一行 Python 语法的声明。例如 "
    'experiment(name="recent_trial", data="recent2", epochs=4, '
    'learning_rate=0.0008, batch_size=128, reason="加快适应")。'
    "仅用 experiment 声明和字面量；在 JSON 字符串中用 \\n 分隔候选。"
)


def parse_strategy(source, data_strategies=DEFAULT_STRATEGIES):
    """静态解释策略声明为训练参数，拒绝其他 Python 语法。"""
    if not isinstance(source, str) or not source.strip() or len(source) > POLICY_LIMIT:
        raise ValueError("策略代码为空或超出长度上限")
    try:
        statements = ast.parse(source, mode="exec").body
    except (SyntaxError, ValueError) as exc:
        raise ValueError("策略代码语法无效") from exc
    if not 1 <= len(statements) <= 4:
        raise ValueError("策略必须声明一至四个候选实验")

    experiments, names = [], set()
    for statement in statements:
        call = statement.value if isinstance(statement, ast.Expr) else None
        if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
                or call.func.id != "experiment" or call.args):
            raise ValueError("策略只接受 experiment(...) 关键字声明")
        params = {}
        for keyword in call.keywords:
            if (keyword.arg not in FIELDS or keyword.arg in params
                    or not isinstance(keyword.value, ast.Constant)):
                raise ValueError("策略参数必须是已声明的字面量")
            params[keyword.arg] = keyword.value.value
        if params.keys() != FIELDS:
            raise ValueError("策略候选缺少必要参数")

        name, data = params["name"], params["data"]
        epochs, rate, batch = params["epochs"], params["learning_rate"], params["batch_size"]
        reason = params["reason"]
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", name)
                or name in names):
            raise ValueError("策略实验名无效或重复")
        if not isinstance(data, str) or data not in data_strategies:
            raise ValueError("策略使用了未授权的数据选择方式")
        if (type(epochs) is not int or not 1 <= epochs <= 8
                or type(rate) not in (int, float) or not 1e-5 <= rate <= 3e-3
                or type(batch) is not int or batch not in (64, 128, 256)):
            raise ValueError("策略超参数超出允许范围")
        if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 240:
            raise ValueError("策略必须提供一条简短理由")
        names.add(name)
        experiments.append(params)
    return experiments


def _reply_object(response):
    if not isinstance(response, str) or len(response) > RESPONSE_LIMIT:
        raise ValueError("模型工具响应类型无效或超出长度限制")
    start = response.find("{")
    stop = response.rfind("}")
    if start < 0 or stop < start:
        raise ValueError("模型工具响应缺少 JSON 对象")
    return json.loads(response[start:stop + 1])


def _save_policy(path, source):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(source)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def request_experiment_plan(chat, prompt, *, strategy_source, history, diagnosis,
                            policy_path=None, audit=None, data_strategies=DEFAULT_STRATEGIES):
    """读取上下文并写策略。旧 experiments JSON 仍可在任一阶段返回。"""
    def emit(action, status, **info):
        if audit is not None:
            audit({"type": "model_tool", "tool": action, "status": status, **info})

    def legacy(reply):
        return "tool" not in reply and isinstance(reply.get("experiments"), list)

    if not isinstance(strategy_source, str) or not 1 <= len(strategy_source) <= 10_000:
        raise ValueError("受信规则源码缺失或超出长度上限")
    messages = [{"role": "user", "content": prompt + FIRST_INSTRUCTION}]
    first = _reply_object(chat(messages, max_tokens=512))
    if legacy(first):
        emit("direct_plan", "legacy")
        return {"experiments": first["experiments"], "tool_protocol": "legacy"}
    if first != {"tool": "read_context", "args": {}}:
        emit("read_context", "rejected")
        raise ValueError("模型必须首先调用无参数的 read_context")

    history = list(history)[-6:]
    context = {"strategy_source": strategy_source, "diagnosis": diagnosis,
               "history": history, "data_strategies": list(data_strategies)}
    context_json = json.dumps(context, ensure_ascii=False)
    while history and len(context_json) > 16_000:
        del history[0]
        context_json = json.dumps(context, ensure_ascii=False)
    if len(context_json) > 16_000:
        emit("read_context", "rejected")
        raise ValueError("策略上下文超出长度上限")
    emit("read_context", "completed", history_count=len(history),
         strategy_sha256=hashlib.sha256(strategy_source.encode("utf-8")).hexdigest())

    messages = messages + [
        {"role": "assistant", "content": json.dumps(first, ensure_ascii=False)},
        {"role": "user", "content": "read_context 工具返回的数据如下。\n" + context_json
         + WRITE_INSTRUCTION},
    ]
    second = _reply_object(chat(messages, max_tokens=1200))
    if legacy(second):
        emit("direct_plan", "legacy")
        return {"experiments": second["experiments"], "tool_protocol": "legacy"}
    args = second.get("args")
    if (second.keys() != {"tool", "args"} or second.get("tool") != "write_strategy"
            or not isinstance(args, dict) or args.keys() != {"code"}):
        emit("write_strategy", "rejected")
        raise ValueError("模型必须使用仅含 code 参数的 write_strategy")

    try:
        experiments = parse_strategy(args["code"], data_strategies)
        source = args["code"].strip() + "\n"
        if policy_path is not None:
            _save_policy(policy_path, source)
    except (ValueError, OSError):
        emit("write_strategy", "rejected")
        raise
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    emit("write_strategy", "completed", policy_sha256=digest,
         artifact=Path(policy_path).name if policy_path is not None else None,
         candidate_names=[exp["name"] for exp in experiments])
    return {"experiments": experiments, "tool_protocol": "policy_dsl", "policy_sha256": digest}
