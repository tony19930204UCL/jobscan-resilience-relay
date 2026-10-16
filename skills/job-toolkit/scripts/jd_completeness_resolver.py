#!/usr/bin/env python3
"""Deterministic full-JD completeness assessment and resolution.

Resolution order is fixed: existing API/ATS text, existing LinkedIn detail fetch,
then read-only public JSON-LD/visible extraction. Sparse evidence is never promoted
to a full JD; the result instead marks browser fallback as required.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


MIN_FULL_JD_CHARS = 700
RESPONSIBILITY_MARKERS = (
    "responsibilities", "what you will do", "what you'll do", "what you’ll do",
    "your role", "the role", "key duties", "main duties", "tasks", "you will",
)
REQUIREMENT_MARKERS = (
    "requirements", "qualifications", "what we are looking for", "what we're looking for",
    "what we’re looking for", "your profile", "who you are", "experience required",
    "skills and experience", "you have", "you bring",
)
API_ATS_HINTS = ("api", "ats", "lever", "greenhouse", "ashby", "linkedin_detail", "jobspy_detail")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def assess_jd_text(text: str, *, claimed_status: str | None = None) -> dict[str, Any]:
    clean = str(text or "").strip()
    normalized = _normalized(clean)
    responsibility_hits = sorted({marker for marker in RESPONSIBILITY_MARKERS if marker in normalized})
    requirement_hits = sorted({marker for marker in REQUIREMENT_MARKERS if marker in normalized})
    reasons: list[str] = []
    if len(clean) < MIN_FULL_JD_CHARS:
        reasons.append("too_short")
    if not responsibility_hits:
        reasons.append("missing_responsibility_section")
    if not requirement_hits:
        reasons.append("missing_requirement_section")
    complete = not reasons
    return {
        "complete": complete,
        "character_count": len(clean),
        "sha256": _sha256_text(clean),
        "claimed_status": claimed_status,
        "responsibility_markers": responsibility_hits,
        "requirement_markers": requirement_hits,
        "reasons": reasons,
        "rule_version": "jd-completeness-v1.0.0",
    }


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"jobhunter_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _default_linkedin_fetcher(job_id: str) -> dict[str, Any]:
    return _load_sibling("job_sources").fetch_linkedin_job_detail(job_id)


def fetch_public_job_text(url: str) -> dict[str, Any]:
    extractor = _load_sibling("public_job_page_extractor")
    fetched = extractor.fetch_public_source(url)
    evidence = extractor.extract_job_text_evidence(fetched.body, content_type=fetched.content_type)
    json_ld = str(evidence.get("json_ld_description") or "").strip()
    visible = str(evidence.get("visible_text") or "").strip()
    if json_ld:
        text, method = json_ld, "json_ld_description"
    else:
        text, method = visible, "visible_text"
    return {
        "text": text,
        "method": method,
        "source_hashes": evidence.get("source_hashes") or {},
        "final_url": fetched.final_url,
        "http_status": fetched.status,
    }


def _existing_method(listing: Mapping[str, Any]) -> str:
    source = " ".join(
        str(listing.get(field) or "")
        for field in ("description_source", "source", "board")
    ).casefold()
    return "existing_api_or_ats_text" if any(hint in source for hint in API_ATS_HINTS) else "existing_full_text"


def resolve_listing_jd(
    listing: Mapping[str, Any],
    *,
    linkedin_fetcher: Callable[[str], Mapping[str, Any]] | None = None,
    public_page_fetcher: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    linkedin_fetcher = linkedin_fetcher or _default_linkedin_fetcher
    public_page_fetcher = public_page_fetcher or fetch_public_job_text
    attempts: list[dict[str, Any]] = []

    existing = str(listing.get("description") or "")
    assessment = assess_jd_text(existing, claimed_status=str(listing.get("detail_status") or ""))
    attempts.append({"method": _existing_method(listing), "assessment": assessment})
    if assessment["complete"]:
        return {
            "status": "FULL_JD_READY",
            "text": existing,
            "method": _existing_method(listing),
            "assessment": assessment,
            "attempts": attempts,
            "browser_fallback_required": False,
        }

    source = str(listing.get("source") or listing.get("board") or "").casefold()
    job_id = str(listing.get("source_job_id") or "").strip()
    if "linkedin" in source and job_id:
        try:
            detail = dict(linkedin_fetcher(job_id))
            text = str(detail.get("description") or "").strip()
            linked_assessment = assess_jd_text(text, claimed_status=str(detail.get("detail_status") or ""))
            attempts.append({"method": "linkedin_detail_fetch", "assessment": linked_assessment})
            if linked_assessment["complete"]:
                return {
                    "status": "FULL_JD_READY",
                    "text": text,
                    "method": "linkedin_detail_fetch",
                    "assessment": linked_assessment,
                    "attempts": attempts,
                    "browser_fallback_required": False,
                }
            assessment = linked_assessment
        except Exception as exc:
            attempts.append({"method": "linkedin_detail_fetch", "error": f"{type(exc).__name__}: {exc}"})

    url = str(listing.get("canonical_url") or listing.get("url") or "").strip()
    if url:
        try:
            public = dict(public_page_fetcher(url))
            text = str(public.get("text") or "").strip()
            public_assessment = assess_jd_text(text)
            public_method = f"public_{public.get('method') or 'visible_text'}"
            attempts.append({
                "method": public_method,
                "assessment": public_assessment,
                "source_hashes": public.get("source_hashes") or {},
            })
            if public_assessment["complete"]:
                return {
                    "status": "FULL_JD_READY",
                    "text": text,
                    "method": public_method,
                    "assessment": public_assessment,
                    "attempts": attempts,
                    "browser_fallback_required": False,
                    "source_hashes": public.get("source_hashes") or {},
                }
            assessment = public_assessment
        except Exception as exc:
            attempts.append({"method": "public_json_ld_or_visible", "error": f"{type(exc).__name__}: {exc}"})

    return {
        "status": "BROWSER_FALLBACK_REQUIRED",
        "text": None,
        "method": None,
        "assessment": assessment,
        "attempts": attempts,
        "browser_fallback_required": True,
    }
