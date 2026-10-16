import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "job-toolkit" / "scripts"
SPEC = importlib.util.spec_from_file_location("daily_job_scan_no_codex_under_test", SCRIPTS / "daily_job_scan_no_codex.py")
orch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = orch
SPEC.loader.exec_module(orch)

TODAY = date.today().isoformat()
DIGEST = "DIGEST-BODY: 1 verified candidate\n"


def script_name(command):
    return Path(command[1]).name


class DailyScanResilienceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.workspace = root / "workspace"
        self.listings = self.workspace / "data" / "listings"
        self.routing = self.workspace / "data" / "jev" / "daily_routing"
        self.logs = self.workspace / "data" / "jev" / "no_codex_daily_scan"
        self.listings.mkdir(parents=True)
        self.pipeline = self.workspace / "data" / "pipeline.json"
        self.pipeline.write_text('{"applications": [], "stats": {}}\n', encoding="utf-8")
        self.pipeline_hash = hashlib.sha256(self.pipeline.read_bytes()).hexdigest()
        self.calls = []
        for name, value in {
            "WORKSPACE": self.workspace,
            "LISTINGS": self.listings,
            "ROUTING_ROOT": self.routing,
            "LOG_ROOT": self.logs,
        }.items():
            patcher = mock.patch.object(orch, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def completed(self, command, stdout="", stderr="", returncode=0):
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)

    def install(self, *, scan="ok", extraction="ok", digest="ok", routing="ok"):
        def fake_run(command, **kwargs):
            name = script_name(command)
            self.calls.append(name)
            behavior = {
                "scan_jobs.py": scan,
                "openrouter_jd_extraction_shadow.py": extraction,
                "make_daily_digest.py": digest,
                "daily_routing_rerun.py": routing,
            }[name]
            if behavior == "timeout":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=f"partial-{name}-stdout".encode(), stderr=f"partial-{name}-stderr".encode())
            if behavior == "fail":
                return self.completed(command, stdout="", stderr=f"failed-{name}", returncode=7)
            if name == "scan_jobs.py":
                (self.listings / f"{TODAY}.json").write_text(json.dumps({"date": TODAY, "listings": []}), encoding="utf-8")
                return self.completed(command, stdout="scan ok\n")
            if name == "make_daily_digest.py":
                return self.completed(command, stdout=DIGEST)
            if name == "daily_routing_rerun.py":
                out = Path(command[command.index("--output-dir") + 1])
                out.mkdir(parents=True, exist_ok=True)
                (out / "compact_report.json").write_text(json.dumps({"selected_count": 0, "jev_accepted": 0, "agy_accepted": 0, "cases": []}), encoding="utf-8")
            return self.completed(command, stdout="ok\n")

        patcher = mock.patch.object(orch.subprocess, "run", side_effect=fake_run)
        patcher.start()
        self.addCleanup(patcher.stop)

    def pipeline_unchanged(self):
        self.assertEqual(hashlib.sha256(self.pipeline.read_bytes()).hexdigest(), self.pipeline_hash)

    def test_extraction_timeout_still_delivers_degraded_digest_and_logs(self):
        self.install(extraction="timeout")
        text = orch.run(reuse_latest=False, skip_agy=True, limit=12)
        self.assertIn(DIGEST.strip(), text)
        lowered = text.lower()
        self.assertIn("extraction", lowered)
        self.assertIn("timeout", lowered)
        self.assertIn("degraded", lowered)
        self.assertIn("Safety boundary", text)
        log_dir = self.logs / TODAY
        self.assertIn("partial-openrouter_jd_extraction_shadow.py-stdout", (log_dir / "02-extraction-shadow.stdout.txt").read_text())
        self.assertIn("partial-openrouter_jd_extraction_shadow.py-stderr", (log_dir / "02-extraction-shadow.stderr.txt").read_text())
        status = json.loads((log_dir / "02-extraction-shadow.json").read_text())
        self.assertIn("timeout", json.dumps(status).lower())
        reviewed = (self.listings / f"{TODAY}-no-codex-reviewed.md").read_text()
        self.assertIn(DIGEST.strip(), reviewed)
        self.assertIn("degraded", reviewed.lower())
        self.assertIn("make_daily_digest.py", self.calls)
        self.pipeline_unchanged()

    def test_nonzero_extraction_remains_advisory_and_is_reported(self):
        self.install(extraction="fail")
        text = orch.run(reuse_latest=False, skip_agy=True, limit=12)
        self.assertIn(DIGEST.strip(), text)
        self.assertIn("degraded", text.lower())
        self.assertIn("7", text)
        self.pipeline_unchanged()

    def test_routing_timeout_is_advisory(self):
        self.install(routing="timeout")
        text = orch.run(reuse_latest=False, skip_agy=False, limit=12)
        self.assertIn(DIGEST.strip(), text)
        lowered = text.lower()
        self.assertIn("routing", lowered)
        self.assertIn("timeout", lowered)
        self.assertIn("degraded", lowered)
        self.assertTrue((self.logs / TODAY / "04-jev-agy-shadow.stderr.txt").is_file())
        self.pipeline_unchanged()

    def test_successful_run_is_not_labelled_degraded(self):
        self.install()
        text = orch.run(reuse_latest=False, skip_agy=False, limit=12)
        self.assertIn(DIGEST.strip(), text)
        self.assertNotIn("degraded", text.lower())
        self.pipeline_unchanged()

    def test_required_scan_timeout_fails_without_a_digest_but_keeps_logs(self):
        self.install(scan="timeout")
        with self.assertRaises(Exception):
            orch.run(reuse_latest=False, skip_agy=True, limit=12)
        self.assertNotIn("make_daily_digest.py", self.calls)
        self.assertIn("partial-scan_jobs.py-stderr", (self.logs / TODAY / "01-scan.stderr.txt").read_text())
        self.pipeline_unchanged()

    def test_required_digest_failure_still_fails_closed(self):
        self.install(digest="fail")
        with self.assertRaises(Exception):
            orch.run(reuse_latest=False, skip_agy=True, limit=12)
        self.assertFalse((self.listings / f"{TODAY}-no-codex-reviewed.md").exists())
        self.pipeline_unchanged()

    def test_main_prints_degraded_digest_and_exits_zero_when_only_advisory_stage_times_out(self):
        self.install(extraction="timeout")
        with mock.patch("builtins.print") as printed:
            code = orch.main(["--skip-agy"])
        self.assertEqual(code, 0)
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list if call.args)
        self.assertIn(DIGEST.strip(), output)
        self.assertNotIn("failed before producing a verified digest", output)
        self.pipeline_unchanged()

    def test_main_required_scan_timeout_exits_nonzero_with_failure_message(self):
        self.install(scan="timeout")
        with mock.patch("builtins.print") as printed:
            code = orch.main(["--skip-agy"])
        self.assertEqual(code, 1)
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list if call.args)
        self.assertIn("failed before producing a verified digest", output)
        self.assertNotIn(DIGEST.strip(), output)
        self.pipeline_unchanged()


if __name__ == "__main__":
    unittest.main()
