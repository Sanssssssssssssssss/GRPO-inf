from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


VALID_REVIEW_MODES = ("extract", "review", "repair")
VALID_EVIDENCE_TYPES = (
    "invoice",
    "purchase_order",
    "goods_receipt",
    "vendor_record",
    "duplicate_payment_check",
    "process_log",
    "clear_invoice_event",
    "payment_terms",
    "policy_excerpt",
    "bpi_event_log",
    "user_statement",
    "unknown",
)
VALID_CREDIBILITY = ("low", "medium", "high")
VALID_SOURCE_TRACEABILITY = (
    "original_document",
    "system_export",
    "log_excerpt",
    "user_statement",
    "rag_guidance",
    "unclear",
)
VALID_SUPPORT_LEVELS = ("none", "partial", "full")
VALID_FIELD_STATUS = ("present", "missing", "conflict", "unclear")
VALID_CONFIDENCE = ("low", "medium", "high")
VALID_CONFLICT_SEVERITY = ("low", "medium", "high")


JSON_VALUE_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {"type": "string"},
        {"type": "number"},
        {"type": "integer"},
        {"type": "boolean"},
        {"type": "null"},
        {"type": "array"},
        {"type": "object"},
    ]
}


EXTRACTED_FIELD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["value", "status", "source_quote", "source_locator", "confidence"],
    "properties": {
        "value": JSON_VALUE_SCHEMA,
        "status": {"enum": list(VALID_FIELD_STATUS)},
        "source_quote": {"type": "string"},
        "source_locator": {"type": "string"},
        "confidence": {"enum": list(VALID_CONFIDENCE)},
    },
}


SUPPORT_RECORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["requirement", "support_level", "quoted_text"],
    "properties": {
        "requirement": {"type": "string"},
        "support_level": {"enum": list(VALID_SUPPORT_LEVELS)},
        "quoted_text": {"type": "string"},
    },
}


CONFLICT_RECORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "type": {"type": ["string", "null"]},
        "conflict_type": {"type": ["string", "null"]},
        "requirement": {"type": ["string", "null"]},
        "severity": {"enum": [*VALID_CONFLICT_SEVERITY, None]},
        "field": {"type": ["string", "null"]},
        "description": {"type": ["string", "null"]},
        "details": {"type": ["string", "null"]},
        "quoted_text": {"type": ["string", "null"]},
        "conflict_with": {"type": ["string", "null"]},
        "compared_to": JSON_VALUE_SCHEMA,
        "required_follow_up": {"type": ["string", "null"]},
        "affected_fields": {"type": "array", "items": {"type": "string"}},
        "affected_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "involved_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "source_values": {"type": ["array", "object", "null"]},
        "suggested_resolution": {"type": ["string", "null"]},
    },
}


EVIDENCE_CARD_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": True}


CASE_UPDATES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary",
        "conversation_summary",
        "case_profile",
        "requirements",
        "remove_requirements",
        "add_evidence",
        "evidence_items",
        "risk_flags",
        "next_questions",
        "next_action_hint",
        "reply_brief",
        "evidence_cards",
    ],
    "properties": {
        "summary": {"type": ["string", "null"]},
        "conversation_summary": {"type": ["string", "null"]},
        "case_profile": {"type": ["object", "null"]},
        "requirements": {"type": "array"},
        "remove_requirements": {"type": "array", "items": {"type": "string"}},
        "add_evidence": {"type": "array"},
        "evidence_items": {"type": "array"},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "next_questions": {"type": "array", "items": {"type": "string"}},
        "next_action_hint": {"type": ["string", "null"]},
        "reply_brief": {"type": ["string", "null"]},
        "evidence_cards": {"type": ["array", "null"], "items": EVIDENCE_CARD_SCHEMA},
    },
}


EVIDENCE_REVIEW_RESULT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "EvidenceReviewResult",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "mode",
        "source_doc_id",
        "evidence_type",
        "credibility",
        "extracted_fields",
        "extraction_result",
        "source_traceability",
        "support_level",
        "risk_flags",
        "should_accept",
        "reason",
        "supports",
        "conflicts",
        "evidence_cards",
        "suggested_patch",
        "reply_to_user",
    ],
    "properties": {
        "mode": {"enum": list(VALID_REVIEW_MODES)},
        "source_doc_id": {"type": "string"},
        "evidence_type": {"enum": list(VALID_EVIDENCE_TYPES)},
        "credibility": {"enum": list(VALID_CREDIBILITY)},
        "extracted_fields": {
            "type": "object",
            "additionalProperties": {
                "anyOf": [
                    EXTRACTED_FIELD_SCHEMA,
                    JSON_VALUE_SCHEMA,
                ]
            },
        },
        "extraction_result": {"type": "object"},
        "source_traceability": {"enum": list(VALID_SOURCE_TRACEABILITY)},
        "support_level": {"enum": list(VALID_SUPPORT_LEVELS)},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "should_accept": {"type": "boolean"},
        "reason": {"type": "string"},
        "supports": {"type": "array", "items": SUPPORT_RECORD_SCHEMA},
        "conflicts": {"type": "array", "items": CONFLICT_RECORD_SCHEMA},
        "evidence_cards": {"type": "array", "items": EVIDENCE_CARD_SCHEMA},
        "suggested_patch": CASE_UPDATES_SCHEMA,
        "reply_to_user": {"type": "string"},
    },
}


REVIEWER_ANSWER_SCHEMA = EVIDENCE_REVIEW_RESULT_SCHEMA
SCHEMA_VALIDATOR = Draft202012Validator(REVIEWER_ANSWER_SCHEMA)


def normalize_completion_text(completion: Any) -> str:
    """Accept TRL standard or conversational completion shapes."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return json.dumps(completion, ensure_ascii=False)
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    parts.extend(str(block.get("text", "")) for block in content if isinstance(block, dict))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(completion)


def strip_wrappers(text: str) -> tuple[str, list[str]]:
    errors: list[str] = []
    value = text.strip()
    if re.search(r"<\|?think\|?>|</\|?think\|?>|<think>|</think>", value, re.IGNORECASE):
        errors.append("thought_tag_leak")
        value = re.sub(r"(?is)<\|?think\|?>.*?</\|?think\|?>", "", value).strip()
        value = re.sub(r"(?is)<think>.*?</think>", "", value).strip()
    if value.startswith("```"):
        errors.append("markdown_fence")
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE).strip()
        value = re.sub(r"\s*```$", "", value).strip()
    return value, errors


def parse_reviewer_json(completion: Any) -> tuple[dict[str, Any] | None, list[str], str]:
    raw = normalize_completion_text(completion)
    cleaned, errors = strip_wrappers(raw)
    try:
        parsed = json.loads(cleaned)
    except Exception as exc:  # pragma: no cover - exact parser text varies
        return None, [*errors, f"json_parse_error:{exc}"], cleaned
    if not isinstance(parsed, dict):
        return None, [*errors, "json_not_object"], cleaned
    return parsed, errors, cleaned


def schema_errors(obj: dict[str, Any]) -> list[str]:
    errors = sorted(SCHEMA_VALIDATOR.iter_errors(obj), key=lambda err: list(err.path))
    return [err.message for err in errors]


def schema_valid(obj: dict[str, Any]) -> bool:
    return not schema_errors(obj)


def partial_schema_score(obj: dict[str, Any]) -> float:
    required = REVIEWER_ANSWER_SCHEMA["required"]
    score = sum(1 for key in required if key in obj) / len(required) * 0.45
    score += 0.10 if obj.get("mode") in VALID_REVIEW_MODES else 0.0
    score += 0.08 if obj.get("evidence_type") in VALID_EVIDENCE_TYPES else 0.0
    score += 0.08 if obj.get("credibility") in VALID_CREDIBILITY else 0.0
    score += 0.08 if obj.get("source_traceability") in VALID_SOURCE_TRACEABILITY else 0.0
    score += 0.08 if obj.get("support_level") in VALID_SUPPORT_LEVELS else 0.0
    score += 0.05 if isinstance(obj.get("extracted_fields"), dict) else 0.0
    score += 0.04 if isinstance(obj.get("supports"), list) else 0.0
    score += 0.04 if isinstance(obj.get("conflicts"), list) else 0.0
    return max(0.0, min(1.0, score))


def write_schema(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(REVIEWER_ANSWER_SCHEMA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
