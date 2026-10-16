# M1 task: advisory stages must not delete the daily digest

Edit only `skills/job-toolkit/scripts/daily_job_scan_no_codex.py` and the mailbox report. Do not edit tests, CI, README or other scripts. No merge, deploy, scheduler or secret changes.

## Current defect
`_run()` lets `subprocess.TimeoutExpired` escape, and logs are written only after `subprocess.run` returns. A timeout in the advisory JD-extraction shadow therefore aborts the run before the digest, with no stage logs.

## Required behaviour
Required stages: collection (`scan_jobs.py`), saved listing validation, digest (`make_daily_digest.py`). They fail closed. A required timeout/nonzero/missing output raises; the CLI prints the existing failure message and exits 1. Required-stage logs must still be written on timeout.

Advisory stages: `openrouter_jd_extraction_shadow.py` and `daily_routing_rerun.py`. Timeout or nonzero exit must not prevent a digest. Catch `TimeoutExpired`, decode partial `output`/`stderr` (bytes, text or None), write the same `.stdout.txt`, `.stderr.txt` and `.json` stage logs, and record status (including timeout) in the JSON log. Continue.

Delivered text and the saved `{run_date}-no-codex-reviewed.md` must include a clearly labelled degraded notice listing each failed/timed-out advisory stage and exit/timeout details. A fully successful run must not contain the word "degraded" anywhere. Keep the safety-boundary footer. Do not change `pipeline.json` or any employer-facing state.

## Verification the owner will run
`python -m pytest -q -o addopts= tests`, plus `python -m py_compile` on touched scripts. Do not weaken or modify tests. Real network/provider calls are not part of this task.

Report schema-v2 JSON in `docs/engineering/HANDOFF.json` as instructed by the relay prompt.
