from __future__ import annotations

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


AP_RISK_ABLATION_SCHEMA: dict[str, Any] = {
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


AP_RISK_ABLATION_VALIDATOR = Draft202012Validator(AP_RISK_ABLATION_SCHEMA)
