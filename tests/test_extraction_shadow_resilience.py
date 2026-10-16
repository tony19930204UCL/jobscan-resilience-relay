import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "job-toolkit" / "scripts"
SPEC = importlib.util.spec_from_file_location("extraction_shadow_under_test", SCRIPTS / "openrouter_jd_extraction_shadow.py")
shadow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = shadow
SPEC.loader.exec_module(shadow)


def make_cases(count=20, suffix=""):
    return [
        {
            "case_id": f"c{i}",
            "title": f"Title {i}",
            "company": "Co",
            "location": "Loc",
            "description_sha256": f"h{i}{suffix}",
            "deterministic_assessment": {"complete": True, "reasons": []},
            "claimed_detail_status": None,
        }
        for i in range(count)
    ]


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def fake_validate(parsed, cases):
    return {"items": [{"case_id": c["case_id"], "accepted": True, "jd_complete": True} for c in cases]}


class ShadowResilienceTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.cases = make_cases()
        self.remote_calls = []
        self.local_calls = []
        for name, value in {
            "choose_candidates": lambda listings, **kw: list(self.cases),
            "build_messages": lambda cases, schema: [],
            "validate_output": fake_validate,
        }.items():
            patcher = mock.patch.object(shadow, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def remote(self, behaviours, step=100.0):
        """behaviours: iterable of 'ok'/'fail', repeated by last value."""
        seq = list(behaviours)

        def runner(model, messages, timeout):
            index = len(self.remote_calls)
            self.remote_calls.append({"model": model, "timeout": timeout, "at": self.clock.now})
            self.clock.now += step
            mode = seq[min(index, len(seq) - 1)]
            if mode == "fail":
                raise TimeoutError("provider timeout")
            return {"parsed": {}, "model": model, "provider": "stub"}

        return runner

    def local(self):
        def runner(model, messages, schema, timeout):
            self.local_calls.append(model)
            return {"parsed": {}, "model": model, "provider": "ollama"}

        return runner

    def call(self, remote, local=None, **kwargs):
        return shadow.run_shadow(
            {"listings": [{}]}, run_date="2026-10-16", limit=20, batch_size=8, timeout=180,
            remote_runner=remote, local_runner=local or self.local(), clock=self.clock, **kwargs,
        )

    def test_default_behaviour_is_complete_and_not_partial(self):
        report = self.call(self.remote(["ok"]), deadline_seconds=None)
        self.assertEqual(report["accepted_case_count"], 20)
        self.assertTrue(report["all_batches_accepted"])
        self.assertFalse(report["partial"])
        self.assertIsNone(report["stop_reason"])
        self.assertEqual(len(self.remote_calls), 3)

    def test_deadline_stops_work_and_marks_report_partial(self):
        report = self.call(self.remote(["fail"], step=100), deadline_seconds=250, breaker_threshold=999)
        self.assertTrue(report["partial"])
        self.assertEqual(report["stop_reason"], "deadline_exceeded")
        self.assertFalse(report["all_batches_accepted"])
        self.assertTrue(all(call["at"] < 250 for call in self.remote_calls))
        self.assertEqual(len(self.remote_calls), 3)
        self.assertEqual(len(report["batches"]), 3)
        self.assertIn("deadline_exceeded", json.dumps(report["batches"]))

    def test_each_call_timeout_is_capped_by_remaining_deadline(self):
        self.call(self.remote(["ok"], step=10), deadline_seconds=50)
        self.assertLessEqual(self.remote_calls[0]["timeout"], 50)
        self.assertLessEqual(self.remote_calls[1]["timeout"], 40)

    def test_breaker_opens_after_consecutive_remote_failures_and_local_still_runs(self):
        report = self.call(self.remote(["fail"], step=1), breaker_threshold=3)
        self.assertEqual(len(self.remote_calls), 3)
        self.assertEqual(report["breakers"]["openrouter"], "open")
        self.assertIn("breaker_open", json.dumps(report["batches"]))
        self.assertEqual(report["accepted_case_count"], 20)
        self.assertEqual(len(self.local_calls), 3)

    def test_success_resets_consecutive_failure_count(self):
        self.cases = make_cases(16)
        report = self.call(self.remote(["fail", "fail", "ok", "fail", "fail", "ok"], step=1), breaker_threshold=3)
        self.assertEqual(len(self.remote_calls), 6)
        self.assertEqual(report["breakers"]["openrouter"], "closed")
        self.assertEqual(report["accepted_case_count"], 16)

    def test_checkpoint_resume_reruns_only_unfinished_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "shadow.checkpoint.jsonl"
            first = self.call(self.remote(["ok"], step=100), deadline_seconds=100, checkpoint_path=checkpoint)
            self.assertTrue(first["partial"])
            self.assertEqual(first["accepted_case_count"], 8)
            self.assertTrue(checkpoint.is_file())
            self.remote_calls.clear()
            second = self.call(self.remote(["ok"], step=1), checkpoint_path=checkpoint)
            self.assertEqual(len(self.remote_calls), 2)
            self.assertEqual(second["accepted_case_count"], 20)
            self.assertEqual(second["reused_batch_count"], 1)
            self.assertFalse(second["partial"])
            self.assertEqual(sorted(item["case_id"] for item in second["results"]), sorted(c["case_id"] for c in self.cases))

    def test_checkpoint_not_reused_when_description_hash_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "shadow.checkpoint.jsonl"
            self.call(self.remote(["ok"], step=1), checkpoint_path=checkpoint)
            self.remote_calls.clear()
            self.cases = make_cases(suffix="-changed")
            report = self.call(self.remote(["ok"], step=1), checkpoint_path=checkpoint)
            self.assertEqual(len(self.remote_calls), 3)
            self.assertEqual(report["reused_batch_count"], 0)

    def test_corrupt_checkpoint_lines_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "shadow.checkpoint.jsonl"
            checkpoint.write_text('{"v": 1, "batch_key": "x"\nnot json at all\n', encoding="utf-8")
            report = self.call(self.remote(["ok"], step=1), checkpoint_path=checkpoint)
            self.assertEqual(report["accepted_case_count"], 20)
            self.assertEqual(report["reused_batch_count"], 0)


class MainWiringTests(unittest.TestCase):
    def run_main(self, extra):
        captured = {}

        def fake_run_shadow(payload, **kwargs):
            captured.update(kwargs)
            return {
                "schema_version": 1, "run_date": "d", "selected_case_count": 1, "accepted_case_count": 0,
                "all_batches_accepted": False, "partial": True, "stop_reason": "deadline_exceeded",
                "deterministic_disagreements": [], "results": [],
            }

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.json"
            out = Path(tmp) / "out" / "report.json"
            src.write_text(json.dumps({"listings": []}), encoding="utf-8")
            with mock.patch.object(shadow, "run_shadow", fake_run_shadow), mock.patch.object(shadow, "load_env", lambda: None):
                code = shadow.main(["--input", str(src), "--output", str(out), *extra])
            return code, captured, out.is_file()

    def test_default_deadline_is_below_the_orchestrator_720s_limit(self):
        code, kwargs, wrote = self.run_main([])
        self.assertEqual(code, 2)
        self.assertTrue(wrote)
        self.assertIsNotNone(kwargs["deadline_seconds"])
        self.assertLessEqual(kwargs["deadline_seconds"], 600)

    def test_cli_options_are_passed_through(self):
        _, kwargs, _ = self.run_main(["--deadline", "123", "--breaker-threshold", "2", "--checkpoint", "/tmp/x.jsonl"])
        self.assertEqual(kwargs["deadline_seconds"], 123)
        self.assertEqual(kwargs["breaker_threshold"], 2)
        self.assertEqual(str(kwargs["checkpoint_path"]), "/tmp/x.jsonl")


if __name__ == "__main__":
    unittest.main()
