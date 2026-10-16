#!/usr/bin/env python3
"""Run the daily Jobhunter scan without invoking the Hermes/Codex agent.

The script owns only collection, deterministic/validated triage, Jev shadow
classification, and bounded AGY advisory review. It never changes pipeline
status and never applies, uploads, messages an employer, or submits anything.
Stdout is a Telegram-ready digest for a no-agent Hermes cron job.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

WORKSPACE = Path(os.environ.get("JOBHUNTER_WORKSPACE", str(Path.cwd())))
SCRIPTS = WORKSPACE / "skills/job-toolkit/scripts"
LISTINGS = WORKSPACE / "data/listings"
ROUTING_ROOT = WORKSPACE / "data/jev/daily_routing"
LOG_ROOT = WORKSPACE / "data/jev/no_codex_daily_scan"
JOB_PYTHON = Path(os.environ.get("JOBHUNTER_PYTHON", sys.executable))


def _run(command: list[str], *, timeout: int, log_stem: Path) -> subprocess.CompletedProcess[str]:
    started = datetime.now().astimezone().isoformat()
    proc = subprocess.run(
        command,
        cwd=WORKSPACE,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    log_stem.parent.mkdir(parents=True, exist_ok=True)
    log_stem.with_suffix(".stdout.txt").write_text(proc.stdout, encoding="utf-8")
    log_stem.with_suffix(".stderr.txt").write_text(proc.stderr, encoding="utf-8")
    log_stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "started_at": started,
                "finished_at": datetime.now().astimezone().isoformat(),
                "command": command,
                "returncode": proc.returncode,
                "stdout_bytes": len(proc.stdout.encode("utf-8")),
                "stderr_bytes": len(proc.stderr.encode("utf-8")),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return proc


def _latest_listing() -> Path:
    files = sorted(LISTINGS.glob("20??-??-??.json"), key=lambda path: path.name)
    if not files:
        raise FileNotFoundError("no dated listing file exists")
    return files[-1]


def _fresh_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(2, 100):
        candidate = Path(f"{base}-r{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot allocate output directory beside {base}")


def _fresh_file(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(2, 100):
        candidate = base.with_name(f"{base.stem}-r{index}{base.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot allocate output file beside {base}")


def _routing_health(run_dir: Path | None) -> list[str]:
    if run_dir is None:
        return ["- Jev/AGY shadow review: not run"]
    report_path = run_dir / "compact_report.json"
    if not report_path.is_file():
        return [f"- Jev/AGY shadow review: failed before compact report (`{run_dir}`)"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fallback_count = 0
    selected_models: dict[str, int] = {}
    for case in report.get("cases") or []:
        model = str(case.get("agy_model") or "none")
        selected_models[model] = selected_models.get(model, 0) + 1
        if case.get("agy_fallback_reason"):
            fallback_count += 1
    models = ", ".join(f"{name}: {count}" for name, count in sorted(selected_models.items())) or "none"
    return [
        f"- Jev shadow accepted: {report.get('jev_accepted', 0)}/{report.get('selected_count', 0)}",
        f"- AGY advisory accepted: {report.get('agy_accepted', 0)}/{report.get('selected_count', 0)}",
        f"- AGY selected models: {models}; secondary-review triggers: {fallback_count}",
        "- Hermes/Codex agent calls for this scheduled digest: **0**",
        "- AGY/Jev outputs remain advisory; no automatic Skip or pipeline mutation is authorised.",
    ]


def run(*, reuse_latest: bool, skip_agy: bool, limit: int) -> str:
    today = date.today().isoformat()
    log_dir = LOG_ROOT / today
    log_dir.mkdir(parents=True, exist_ok=True)

    scan_note = None
    if reuse_latest:
        listing_path = _latest_listing()
        scan_note = f"Pilot/replay mode: reused `{listing_path.name}`; no live collection was run."
    else:
        scan = _run(
            [str(JOB_PYTHON), str(SCRIPTS / "scan_jobs.py")],
            timeout=1500,
            log_stem=log_dir / "01-scan",
        )
        expected = LISTINGS / f"{today}.json"
        if scan.returncode != 0 or not expected.is_file():
            detail = f"scan_jobs.py exit {scan.returncode}; expected output present={expected.is_file()}"
            raise RuntimeError(detail)
        listing_path = expected

    payload = json.loads(listing_path.read_text(encoding="utf-8"))
    run_date = str(payload.get("date") or listing_path.stem)

    extraction_output = _fresh_file(ROUTING_ROOT / f"openrouter-jd-extraction-shadow-{run_date}.json")
    extraction = _run(
        [
            str(JOB_PYTHON),
            str(SCRIPTS / "openrouter_jd_extraction_shadow.py"),
            "--input",
            str(listing_path),
            "--output",
            str(extraction_output),
            "--run-date",
            run_date,
            "--limit",
            "20",
        ],
        timeout=720,
        log_stem=log_dir / "02-extraction-shadow",
    )

    digest = _run(
        [
            str(JOB_PYTHON),
            str(SCRIPTS / "make_daily_digest.py"),
            "--input",
            str(listing_path),
            "--limit",
            str(limit),
            "--no-second-pass",
        ],
        timeout=720,
        log_stem=log_dir / "03-digest",
    )
    if digest.returncode != 0 or not digest.stdout.strip():
        raise RuntimeError(f"make_daily_digest.py failed with exit {digest.returncode}")

    routing_dir: Path | None = None
    if not skip_agy:
        routing_dir = _fresh_dir(ROUTING_ROOT / f"{run_date}-no-codex-shadow")
        routing = _run(
            [
                str(JOB_PYTHON),
                str(SCRIPTS / "daily_routing_rerun.py"),
                "--input",
                str(listing_path),
                "--run-date",
                run_date,
                "--output-dir",
                str(routing_dir),
                "--limit",
                "3",
                "--run-agy",
                "--agy-model",
                "claude-sonnet-4-6",
                "--no-agy-fallback",
            ],
            timeout=900,
            log_stem=log_dir / "04-jev-agy-shadow",
        )
        if routing.returncode != 0:
            routing_dir = None

    lines = [digest.stdout.strip(), "", "## Routing health"]
    lines.extend(_routing_health(routing_dir))
    lines.append(f"- OpenRouter JD extraction shadow exit: {extraction.returncode}; artifact: `{extraction_output}`")
    if scan_note:
        lines.append(f"- {scan_note}")
    lines.extend(
        [
            "",
            "**Safety boundary:** this run did not apply, upload, message, submit, or alter the application pipeline.",
        ]
    )
    final = "\n".join(lines).strip() + "\n"
    reviewed = LISTINGS / f"{run_date}-no-codex-reviewed.md"
    reviewed.write_text(final, encoding="utf-8")
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-latest", action="store_true", help="Test against the newest saved listing file instead of collecting live data")
    parser.add_argument("--skip-agy", action="store_true", help="Skip Jev/AGY advisory review")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args(argv)
    try:
        print(run(reuse_latest=args.reuse_latest, skip_agy=args.skip_agy, limit=args.limit), end="")
        return 0
    except Exception as exc:
        print(
            "⚠️ Daily job scan failed before producing a verified digest.\n\n"
            f"Reason: `{type(exc).__name__}: {exc}`\n\n"
            f"Logs: `{LOG_ROOT / date.today().isoformat()}`\n"
            "No application, message, upload, submission, or pipeline change occurred."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
