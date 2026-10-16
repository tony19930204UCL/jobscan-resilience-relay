#!/usr/bin/env python3
"""OpenRouter-first shadow extraction for heterogeneous job descriptions.

This lane is read-only and advisory. It preserves normalized listing metadata,
asks a remote free model to identify JD completeness with exact source spans,
validates every span locally, and falls back across models. It never authorizes
Skip, Apply, Save, Submit, messaging, upload, or pipeline mutation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import requests

SCHEMA_VERSION = "jobhunter-openrouter-jd-extraction-shadow-v1.0.0"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PRIMARY_MODEL = "qwen/qwen3.8-27b:free"
REMOTE_FALLBACK_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
REMOTE_TERTIARY_MODEL = "nex-agi/nex-n2.5-pro:free"
LOCAL_FALLBACK_MODEL = "qwen3.5:9b"
ENV_FILES = tuple(Path(item) for item in os.environ.get("JOBHUNTER_ENV_FILES", "").split(os.pathsep) if item)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_env() -> None:
    for path in ENV_FILES:
        if not path.is_file():
            continue
        for raw in path.read_text(errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_sibling(name: str):
    import importlib.util
    import sys

    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"openrouter_shadow_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def model_schema() -> dict[str, Any]:
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "case_id": {"type": "string"},
            "jd_complete": {"type": "boolean"},
            "responsibility_quotes": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
            },
            "qualification_quotes": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
            },
            "language_hint": {"type": ["string", "null"]},
            "short_reason": {"type": "string"},
        },
        "required": [
            "case_id",
            "jd_complete",
            "responsibility_quotes",
            "qualification_quotes",
            "language_hint",
            "short_reason",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"items": {"type": "array", "items": item}},
        "required": ["items"],
    }


def build_messages(cases: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]) -> list[dict[str, str]]:
    system = """You are an exact multilingual job-description extraction engine, not a career adviser.
For every case:
1. Analyze only description_text. Metadata fields are context and identifiers, not evidence that the JD is complete.
2. jd_complete=true only when description_text itself contains BOTH at least one concrete job responsibility/duty AND at least one concrete candidate qualification/requirement. Employer marketing, project background, benefits, application instructions, and generic opportunity summaries do not substitute.
3. Return 1-3 short responsibility_quotes and 1-3 short qualification_quotes copied character-for-character from description_text. Preserve spelling, punctuation and capitalization, including lowercase when a quote starts mid-sentence. Never paraphrase, translate, reconstruct, or repair a quote.
4. If either category is absent, set jd_complete=false. Return exact quotes only for categories actually present; absent-category quote arrays must be empty.
5. Understand any language and any heading style. Do not rely on English section names.
6. Return exactly one item for every case_id, no duplicate/extra case, and only JSON matching the schema.
Exact output JSON Schema:
""" + stable_json(schema)
    model_cases = [
        {
            "case_id": case["case_id"],
            "title": case.get("title"),
            "company": case.get("company"),
            "location": case.get("location"),
            "description_text": case["description_text"],
        }
        for case in cases
    ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps({"cases": model_cases}, ensure_ascii=False)},
    ]


def _safe_error(exc: Exception) -> str:
    text = re.sub(r"\s+", " ", str(exc)).strip()
    return f"{type(exc).__name__}: {text}"[:500]


def call_openrouter(
    model: str,
    messages: list[dict[str, str]],
    *,
    timeout: float,
    api_key: str | None = None,
) -> dict[str, Any]:
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY missing")
    started = time.perf_counter()
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://hermes-agent.nousresearch.com/",
            "X-Title": "Hermes Jobhunter JD Extraction Shadow",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 4000,
            "reasoning": {"effort": "none", "exclude": True},
        },
        timeout=timeout,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(f"HTTP {response.status_code} non-JSON response") from exc
    if response.status_code >= 400 or not isinstance(payload.get("choices"), list) or not payload["choices"]:
        message = ((payload.get("error") or {}).get("message") if isinstance(payload.get("error"), Mapping) else None) or f"HTTP {response.status_code} missing choices"
        raise RuntimeError(str(message))
    content = payload["choices"][0].get("message", {}).get("content") or ""
    return {
        "model": payload.get("model") or model,
        "provider": payload.get("provider"),
        "elapsed_ms": elapsed_ms,
        "usage": payload.get("usage"),
        "parsed": json.loads(content),
    }


def call_local_ollama(
    model: str,
    messages: list[dict[str, str]],
    schema: Mapping[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    request = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": "2m",
        "format": schema,
        "messages": messages,
        "options": {"temperature": 0, "seed": 42, "num_ctx": 16384},
    }
    windows_dir_value = os.environ.get("JOBHUNTER_WINDOWS_TEMP_WSL")
    windows_dir_native = os.environ.get("JOBHUNTER_WINDOWS_TEMP_WIN")
    if not windows_dir_value or not windows_dir_native:
        raise RuntimeError("local Ollama Windows bridge is not configured")
    windows_dir = Path(windows_dir_value)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False, dir=windows_dir) as handle:
        json.dump(request, handle, ensure_ascii=False)
        input_path = Path(handle.name)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".result.json", delete=False, dir=windows_dir) as handle:
        output_path = Path(handle.name)
    win_input = windows_dir_native.rstrip("\\") + "\\" + input_path.name
    win_output = windows_dir_native.rstrip("\\") + "\\" + output_path.name
    powershell = (
        "$ErrorActionPreference='Stop';"
        f"$body=[IO.File]::ReadAllText('{win_input}',[Text.Encoding]::UTF8);"
        "$sw=[Diagnostics.Stopwatch]::StartNew();"
        "$r=Invoke-RestMethod -Uri 'http://localhost:11434/api/chat' -Method Post -ContentType 'application/json; charset=utf-8' -Body $body;"
        "$sw.Stop();"
        "$out=[pscustomobject]@{model=$r.model;content=$r.message.content;wall_ms=$sw.ElapsedMilliseconds;prompt_eval_count=$r.prompt_eval_count;eval_count=$r.eval_count}|ConvertTo-Json -Compress;"
        f"[IO.File]::WriteAllText('{win_output}',$out,[Text.Encoding]::UTF8)"
    )
    subprocess.run(
        ["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe", "-NoProfile", "-Command", powershell],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    result = json.loads(output_path.read_text(encoding="utf-8-sig"))
    return {
        "model": result.get("model") or model,
        "provider": "ollama",
        "elapsed_ms": result.get("wall_ms"),
        "usage": {
            "prompt_eval_count": result.get("prompt_eval_count"),
            "eval_count": result.get("eval_count"),
        },
        "parsed": json.loads(result.get("content") or ""),
    }


def validate_output(parsed: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {str(case["case_id"]): case for case in cases}
    rows = parsed.get("items") if isinstance(parsed, Mapping) else None
    if not isinstance(rows, list):
        return {"accepted": False, "errors": ["items_not_list"], "items": []}
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    global_errors: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            global_errors.append("non_object_item")
            continue
        case_id = str(row.get("case_id") or "")
        source = expected.get(case_id)
        errors: list[str] = []
        if source is None:
            errors.append("unknown_case_id")
            text = ""
        elif case_id in seen:
            errors.append("duplicate_case_id")
            text = str(source["description_text"])
        else:
            seen.add(case_id)
            text = str(source["description_text"])
        responsibility_quotes = row.get("responsibility_quotes")
        qualification_quotes = row.get("qualification_quotes")
        if not isinstance(row.get("jd_complete"), bool):
            errors.append("jd_complete_not_boolean")
        for field, quotes in (("responsibility", responsibility_quotes), ("qualification", qualification_quotes)):
            if not isinstance(quotes, list) or not all(isinstance(q, str) and q for q in quotes):
                errors.append(f"{field}_quotes_invalid")
                continue
            if len(quotes) > 3:
                errors.append(f"{field}_quotes_too_many")
            for quote in quotes:
                if quote not in text:
                    errors.append(f"{field}_quote_not_verbatim")
        if row.get("jd_complete") is True and (not responsibility_quotes or not qualification_quotes):
            errors.append("complete_without_both_evidence_types")
        if row.get("jd_complete") is False and responsibility_quotes and qualification_quotes:
            errors.append("incomplete_despite_both_evidence_types")
        results.append({
            "case_id": case_id,
            "accepted": not errors,
            "errors": sorted(set(errors)),
            "jd_complete": row.get("jd_complete"),
            "responsibility_quotes": responsibility_quotes if isinstance(responsibility_quotes, list) else [],
            "qualification_quotes": qualification_quotes if isinstance(qualification_quotes, list) else [],
            "language_hint": row.get("language_hint"),
            "short_reason": row.get("short_reason"),
        })
    missing = sorted(set(expected) - seen)
    if missing:
        global_errors.append("missing_cases:" + ",".join(missing))
    accepted = not global_errors and len(results) == len(cases) and all(item["accepted"] for item in results)
    return {"accepted": accepted, "errors": global_errors, "items": results}


def choose_candidates(
    listings: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    run_date: str | None,
) -> list[dict[str, Any]]:
    resolver = load_sibling("jd_completeness_resolver")
    candidates: list[dict[str, Any]] = []
    for row in listings:
        description = str(row.get("description") or "").strip()
        if len(description) < 80:
            continue
        canonical_key = str(row.get("canonical_key") or row.get("url") or "")
        if not canonical_key:
            continue
        deterministic = resolver.assess_jd_text(description, claimed_status=str(row.get("detail_status") or ""))
        history = row.get("listing_history") if isinstance(row.get("listing_history"), Mapping) else {}
        first_seen = str(row.get("first_seen") or history.get("first_seen") or "")
        is_new = bool(run_date and first_seen.startswith(run_date))
        claimed_full = str(row.get("detail_status") or "").casefold() == "full"
        disagreement_priority = claimed_full and not deterministic["complete"]
        candidates.append({
            "case_id": canonical_key,
            "title": row.get("title"),
            "company": row.get("company"),
            "location": row.get("location"),
            "description_text": description,
            "description_sha256": sha256_text(description),
            "score": row.get("score"),
            "first_seen": first_seen or None,
            "is_new": is_new,
            "claimed_detail_status": row.get("detail_status"),
            "deterministic_assessment": deterministic,
            "selection_priority": {
                "claimed_full_but_rule_incomplete": disagreement_priority,
                "new_on_run_date": is_new,
            },
        })
    candidates.sort(
        key=lambda item: (
            not item["selection_priority"]["claimed_full_but_rule_incomplete"],
            not item["is_new"],
            -(float(item["score"]) if isinstance(item.get("score"), (int, float)) else -1),
            str(item.get("case_id")),
        )
    )
    return candidates[:limit]


def run_shadow(
    payload: Mapping[str, Any],
    *,
    run_date: str | None,
    limit: int,
    batch_size: int,
    timeout: float,
    remote_runner: Callable[..., Mapping[str, Any]] | None = None,
    local_runner: Callable[..., Mapping[str, Any]] | None = None,
    deadline_seconds: float | None = None,
    breaker_threshold: int = 3,
    checkpoint_path: Path | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    listings = payload.get("listings")
    if not isinstance(listings, list):
        raise ValueError("payload.listings must be a list")
    cases = choose_candidates(listings, limit=limit, run_date=run_date)
    schema = model_schema()
    remote_runner = remote_runner or call_openrouter
    local_runner = local_runner or call_local_ollama
    started_at = clock()
    failures = {"openrouter": 0, "ollama": 0}
    breaker_threshold = max(1, breaker_threshold)
    partial = False
    stop_reason = None
    reused_batch_count = 0
    checkpoints: dict[str, dict[str, Any]] = {}
    if checkpoint_path is not None and checkpoint_path.is_file():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                if (isinstance(record, dict) and record.get("v") == 1
                        and isinstance(record.get("batch_key"), str)
                        and isinstance(record.get("batch"), dict)
                        and record["batch"].get("accepted") is True
                        and isinstance(record.get("items"), list)):
                    checkpoints[record["batch_key"]] = record
            except (ValueError, TypeError, KeyError):
                continue
    batches: list[dict[str, Any]] = []
    accepted_items: list[dict[str, Any]] = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        batch_key = sha256_text(stable_json(sorted(
            str(case["case_id"]) + ":" + str(case["description_sha256"]) for case in batch
        )))
        saved = checkpoints.get(batch_key)
        if saved is not None and saved["batch"].get("case_ids") == [case["case_id"] for case in batch] and len(saved["items"]) == len(batch):
            batches.append({**saved["batch"], "batch_index": len(batches) + 1})
            accepted_items.extend(saved["items"])
            reused_batch_count += 1
            continue
        if deadline_seconds is not None and clock() - started_at >= deadline_seconds:
            partial = True
            stop_reason = "deadline_exceeded"
            batches.append({
                "batch_index": len(batches) + 1,
                "case_ids": [case["case_id"] for case in batch],
                "accepted": False,
                "unresolved_case_ids": [case["case_id"] for case in batch],
                "attempts": [{"error": "deadline_exceeded", "skipped": True}],
            })
            continue
        batch_items_start = len(accepted_items)
        attempts: list[dict[str, Any]] = []
        accepted_by_id: dict[str, dict[str, Any]] = {}
        model_meta_by_id: dict[str, dict[str, Any]] = {}
        remaining = list(batch)
        for kind, model in (
            ("openrouter", PRIMARY_MODEL),
            ("openrouter", REMOTE_FALLBACK_MODEL),
            ("openrouter_repair", REMOTE_FALLBACK_MODEL),
            ("openrouter_tertiary", REMOTE_TERTIARY_MODEL),
            ("ollama", LOCAL_FALLBACK_MODEL),
        ):
            if not remaining:
                break
            route = "openrouter" if kind.startswith("openrouter") else "ollama"
            elapsed = clock() - started_at
            if deadline_seconds is not None and elapsed >= deadline_seconds:
                partial = True
                stop_reason = "deadline_exceeded"
                attempts.append({"kind": kind, "requested_model": model, "skipped": True, "error": "deadline_exceeded"})
                break
            if failures[route] >= breaker_threshold:
                attempts.append({"kind": kind, "requested_model": model, "skipped": True, "error": "breaker_open"})
                continue
            call_timeout = timeout if deadline_seconds is None else min(timeout, deadline_seconds - elapsed)
            messages = build_messages(remaining, schema)
            attempt_case_ids = [case["case_id"] for case in remaining]
            try:
                if kind.startswith("openrouter"):
                    receipt = dict(remote_runner(model, messages, timeout=call_timeout))
                else:
                    receipt = dict(local_runner(model, messages, schema, timeout=call_timeout))
                failures[route] = 0
                validation = validate_output(receipt.pop("parsed"), remaining)
                attempts.append({
                    "kind": kind,
                    "requested_model": model,
                    "case_ids": attempt_case_ids,
                    **receipt,
                    "validation": validation,
                })
                for item in validation["items"]:
                    if item["accepted"]:
                        accepted_by_id[item["case_id"]] = item
                        model_meta_by_id[item["case_id"]] = {
                            "model": receipt.get("model"),
                            "provider": receipt.get("provider"),
                        }
                remaining = [case for case in remaining if case["case_id"] not in accepted_by_id]
            except Exception as exc:
                failures[route] += 1
                attempts.append({
                    "kind": kind,
                    "requested_model": model,
                    "case_ids": attempt_case_ids,
                    "error": _safe_error(exc),
                })
        batch_accepted = not remaining and len(accepted_by_id) == len(batch)
        if accepted_by_id:
            by_id = {item["case_id"]: item for item in batch}
            for case_id in [case["case_id"] for case in batch if case["case_id"] in accepted_by_id]:
                item = accepted_by_id[case_id]
                source = by_id[case_id]
                meta = model_meta_by_id[case_id]
                accepted_items.append({
                    **item,
                    "title": source.get("title"),
                    "company": source.get("company"),
                    "location": source.get("location"),
                    "description_sha256": source["description_sha256"],
                    "deterministic_complete": source["deterministic_assessment"]["complete"],
                    "deterministic_reasons": source["deterministic_assessment"]["reasons"],
                    "claimed_detail_status": source.get("claimed_detail_status"),
                    "model": meta.get("model"),
                    "provider": meta.get("provider"),
                })
        batches.append({
            "batch_index": len(batches) + 1,
            "case_ids": [case["case_id"] for case in batch],
            "accepted": batch_accepted,
            "unresolved_case_ids": [case["case_id"] for case in remaining],
            "attempts": attempts,
        })
        if batch_accepted and checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"v": 1, "batch_key": batch_key, "batch": batches[-1], "items": accepted_items[batch_items_start:]}
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(stable_json(record) + "\n")
    disagreements = [
        item for item in accepted_items
        if item.get("jd_complete") is not item.get("deterministic_complete")
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "source_listing_count": len(listings),
        "selected_case_count": len(cases),
        "accepted_case_count": len(accepted_items),
        "all_batches_accepted": not partial and len(accepted_items) == len(cases),
        "partial": partial,
        "stop_reason": stop_reason,
        "breakers": {route: "open" if count >= breaker_threshold else "closed" for route, count in failures.items()},
        "reused_batch_count": reused_batch_count,
        "primary_model": PRIMARY_MODEL,
        "remote_fallback_model": REMOTE_FALLBACK_MODEL,
        "remote_tertiary_model": REMOTE_TERTIARY_MODEL,
        "local_emergency_model": LOCAL_FALLBACK_MODEL,
        "selected_cases": cases,
        "results": accepted_items,
        "deterministic_disagreement_count": len(disagreements),
        "deterministic_disagreements": disagreements,
        "batches": batches,
        "shadow_only": True,
        "automatic_skip_authorized": False,
        "pipeline_mutation_authorized": False,
        "external_job_action_authorized": False,
    }


def render_summary(report: Mapping[str, Any], output_path: Path) -> str:
    lines = [
        f"OpenRouter JD extraction shadow — {report.get('run_date') or 'undated'}",
        f"Selected: {report.get('selected_case_count', 0)} | Accepted: {report.get('accepted_case_count', 0)} | Rule disagreements: {report.get('deterministic_disagreement_count', 0)}",
        f"Artifact: {output_path}",
    ]
    for item in report.get("deterministic_disagreements", [])[:10]:
        direction = f"Python={item.get('deterministic_complete')} / LLM={item.get('jd_complete')}"
        lines.append(f"- {item.get('title')} @ {item.get('company')}: {direction}; model={item.get('model')}")
    if not report.get("all_batches_accepted"):
        lines.append("WARNING: at least one model batch failed validation; inspect batch attempts before using results.")
    lines.append("Shadow-only: no automatic Skip, application, message, upload, submission, or pipeline mutation.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-date", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--deadline", type=float, default=540)
    parser.add_argument("--breaker-threshold", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    load_env()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = run_shadow(
        payload,
        run_date=args.run_date,
        limit=max(1, args.limit),
        batch_size=max(1, args.batch_size),
        timeout=args.timeout,
        deadline_seconds=args.deadline,
        breaker_threshold=args.breaker_threshold,
        checkpoint_path=args.checkpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(render_summary(report, args.output))
    return 0 if report["all_batches_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
