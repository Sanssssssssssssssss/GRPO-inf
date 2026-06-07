from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft202012Validator


VALID_DECISIONS = ("approve", "hold", "reject", "escalate")
VALID_RISKS = ("low", "medium", "high", "critical")
VALID_FINDING_TYPES = (
    "amount_mismatch",
    "tax_mismatch",
    "currency_mismatch",
    "quantity_grn_mismatch",
    "missing_po",
    "missing_grn",
    "vendor_mismatch",
    "duplicate_invoice",
    "bank_change",
    "contract_expired",
    "conflicting_email_invoice",
    "prompt_injection",
    "split_invoice",
    "payment_terms_violation",
    "other",
)
VALID_SEVERITIES = ("low", "medium", "high", "critical")


REVIEWER_ANSWER_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "risk_level",
        "findings",
        "missing_evidence",
        "unsupported_items",
        "confidence",
    ],
    "properties": {
        "decision": {"enum": list(VALID_DECISIONS)},
        "risk_level": {"enum": list(VALID_RISKS)},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "type",
                    "severity",
                    "expected",
                    "observed",
                    "source_ids",
                    "evidence_quotes",
                    "recommended_action",
                ],
                "properties": {
                    "type": {"enum": list(VALID_FINDING_TYPES)},
                    "severity": {"enum": list(VALID_SEVERITIES)},
                    "expected": {"type": ["string", "null"]},
                    "observed": {"type": ["string", "null"]},
                    "source_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "evidence_quotes": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "recommended_action": {"type": "string"},
                },
            },
        },
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
        "unsupported_items": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


SCHEMA_VALIDATOR = Draft202012Validator(REVIEWER_ANSWER_SCHEMA)


def normalize_completion_text(completion: Any) -> str:
    """Accept TRL standard or conversational completion shapes."""
    if isinstance(completion, str):
        return completion
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
    score = 0.0
    score += sum(1 for key in required if key in obj) / len(required) * 0.35
    score += 0.15 if obj.get("decision") in VALID_DECISIONS else 0.0
    score += 0.15 if obj.get("risk_level") in VALID_RISKS else 0.0
    score += 0.10 if isinstance(obj.get("findings"), list) else 0.0
    score += 0.10 if isinstance(obj.get("missing_evidence"), list) else 0.0
    score += 0.10 if isinstance(obj.get("unsupported_items"), list) else 0.0
    confidence = obj.get("confidence")
    score += 0.05 if isinstance(confidence, (int, float)) and 0 <= confidence <= 1 else 0.0
    return max(0.0, min(1.0, score))
