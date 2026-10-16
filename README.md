# jobscan-resilience-relay

Reliability work on a personal job-scan orchestrator: required stages (collection, listing persistence, digest) must fail closed; advisory stages (JD extraction shadow, routing/review shadow) may time out without deleting the day's digest.

This repository intentionally contains no CV, candidate evidence, scan data or private ranking prompts. Other scan stages are stubbed in tests. Do not add credentials or personal data.
