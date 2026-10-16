# M2 task: bounded, resumable JD-extraction shadow

Edit only `skills/job-toolkit/scripts/openrouter_jd_extraction_shadow.py` and `docs/engineering/HANDOFF.json`. Do not edit tests, CI or other files. No merge.

## Defect
`run_shadow` tries up to 5 attempts per batch (3 remote models, 1 repair, 1 local) with a 180 s per-call timeout and has no overall deadline: 3 batches can take 2700 s, far above the orchestrator's 720 s stage limit. A dead provider is retried for every batch, and nothing is saved until the end.

## Required (all verified by tests/test_extraction_shadow_resilience.py, read it for exact names)
1. `run_shadow` gains keyword args: `deadline_seconds: float | None = None`, `breaker_threshold: int = 3`, `checkpoint_path: Path | None = None`, `clock: Callable[[], float] = time.monotonic`.
2. Deadline: measured from the start of `run_shadow` using `clock`. Before each attempt, if elapsed >= deadline, do not call any runner; record a skipped attempt or batch with the text `deadline_exceeded`, mark remaining batches unresolved, set `partial=True` and `stop_reason="deadline_exceeded"`. Cap every per-call timeout to `min(timeout, remaining_deadline)`.
3. Breaker: route key `openrouter` for all remote kinds, `ollama` for local. Count consecutive failures (exceptions) per route; any success resets to 0. When the count reaches `breaker_threshold`, remaining attempts on that route, in this and later batches, are recorded as skipped with the text `breaker_open` and no runner call is made. The local route keeps working. Report field `breakers` maps route to `"open"` or `"closed"`.
4. Checkpoint (JSON Lines, only when `checkpoint_path` is given): after each fully accepted batch append one line holding `v`:1, `batch_key` (sha256 of the sorted list of `case_id:description_sha256` for that batch), the batch report entry and its accepted items. On start, read the file, ignore blank, corrupt or unknown-version lines without crashing, and reuse a batch only when its `batch_key` matches exactly (so a changed description is never reused). Reused batches make no runner calls. Report field `reused_batch_count`.
5. Report always includes `partial` (bool), `stop_reason` (None or string), `breakers`, `reused_batch_count`. `all_batches_accepted` must be False when partial. Existing report fields and behaviour are unchanged when none of the new options trigger.
6. CLI: add `--deadline` (float, default 540), `--breaker-threshold` (int, default 3), `--checkpoint` (path, default none). Pass them to `run_shadow` as `deadline_seconds`, `breaker_threshold`, `checkpoint_path`. Always write the output JSON, even when partial, and return exit code 2 when not all batches were accepted. Keep the shadow-only safety flags.
7. Must compile and run on Python 3.11: never reuse the same quote character inside an f-string expression; use real newlines in the source.

Report schema-v2 JSON in `docs/engineering/HANDOFF.json`; report tests as NOT_RUN (the sandbox cannot run them; the owner does).
