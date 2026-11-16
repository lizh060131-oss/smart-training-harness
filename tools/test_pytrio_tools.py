# -*- coding: utf-8 -*-
"""离线验证 PyTRiO 工具桥与训练策略语言。"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1" / "harness_v2"))
from pytrio_tools import parse_strategy, request_experiment_plan  # noqa: E402


CODE = ('experiment(name="new_recent", data="recent2", epochs=4, '
        'learning_rate=0.0008, batch_size=128, reason="强调当前批次")')
READ = json.dumps({"tool": "read_context", "args": {}})
SOURCE = 'def choose():\n    return "full_hist"\n'


class TestTools(unittest.TestCase):
    def test_reads_context_and_writes_same_policy_that_is_evaluated(self):
        write = json.dumps({"tool": "write_strategy", "args": {"code": CODE}})
        chat = Mock(side_effect=[READ, write])
        audit = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plans" / "round_04.py"
            result = request_experiment_plan(
                chat, "plan A", strategy_source=SOURCE,
                history=[{"round": 3, "chosen": "stable"}], diagnosis={"max_psi": 0.7},
                policy_path=path, audit=audit)
            saved = path.read_text(encoding="utf-8")
            self.assertEqual(result["experiments"], parse_strategy(saved))
            self.assertEqual(result["policy_sha256"], hashlib.sha256(saved.encode()).hexdigest())
            self.assertEqual([item.name for item in path.parent.iterdir()], [path.name])

        self.assertEqual(result["tool_protocol"], "policy_dsl")
        self.assertEqual([call.kwargs["max_tokens"] for call in chat.call_args_list], [512, 1200])
        context = chat.call_args_list[1].args[0][-1]["content"]
        self.assertIn('"round": 3', context)
        self.assertIn('"max_psi": 0.7', context)
        self.assertIn("def choose()", context)
        calls = [call.args[0] for call in audit.call_args_list]
        self.assertEqual([call["tool"] for call in calls], ["read_context", "write_strategy"])
        self.assertEqual([call["status"] for call in calls], ["completed", "completed"])
        self.assertEqual(calls[0]["history_count"], 1)

    def test_legacy_plan_still_works_with_one_call(self):
        experiments = parse_strategy(CODE)
        chat = Mock(return_value="```json\n" + json.dumps({"experiments": experiments}) + "\n```")
        audit = Mock()
        result = request_experiment_plan(chat, "plan old", strategy_source=SOURCE,
                                         history=[], diagnosis="stable", audit=audit)
        self.assertEqual(result, {"experiments": experiments, "tool_protocol": "legacy"})
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(audit.call_args.args[0]["status"], "legacy")

    def test_unknown_argument_does_not_write_a_policy(self):
        write = json.dumps({"tool": "write_strategy", "args": {"code": CODE, "path": "extra.py"}})
        chat, audit = Mock(side_effect=[READ, write]), Mock()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "round_01.py"
            with self.assertRaises(ValueError):
                request_experiment_plan(chat, "plan A", strategy_source=SOURCE,
                                        history=[], diagnosis={}, policy_path=path, audit=audit)
            self.assertFalse(path.exists())
        self.assertEqual(audit.call_args.args[0]["status"], "rejected")

    def test_only_bounded_literal_experiment_declarations_are_supported(self):
        for invalid in (
            "import math\n" + CODE,
            CODE.replace("epochs=4", "epochs=2+2"),
            CODE.replace("epochs=4", "epochs=40"),
            CODE.replace('data="recent2"', 'data="other"'),
            CODE.replace("learning_rate=0.0008", "learning_rate=1e999"),
            CODE + "\n" + CODE,
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_strategy(invalid)


if __name__ == "__main__":
    unittest.main()
