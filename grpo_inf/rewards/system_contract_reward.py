from __future__ import annotations

from typing import Any

from grpo_inf.schema import parse_reviewer_json, partial_schema_score, schema_errors


def score_system_contract(completion: Any) -> dict[str, Any]:
    obj, errors, cleaned = parse_reviewer_json(completion)
    if obj is None:
        return {
            "component": 0.0,
            "json_valid": 0.0,
            "schema_valid": 0.0,
            "contract_valid": 0.0,
            "thought_leak": 1.0 if "thought_tag_leak" in errors else 0.0,
            "markdown_fence": 1.0 if "markdown_fence" in errors else 0.0,
            "penalty": -1.0,
            "schema_errors": [],
            "errors": errors,
            "cleaned_completion": cleaned,
            "object": None,
        }

    errors_for_schema = schema_errors(obj)
    schema_score = 1.0 if not errors_for_schema else partial_schema_score(obj)
    penalty = 0.0
    if "thought_tag_leak" in errors:
        penalty -= 0.10
    if "markdown_fence" in errors:
        penalty -= 0.06
    if any(key in obj for key in ("reasoning", "chain_of_thought", "thoughts")):
        penalty -= 0.10
        errors.append("thought_field_leak")

    return {
        "component": schema_score,
        "json_valid": 1.0,
        "schema_valid": 1.0 if not errors_for_schema else 0.0,
        "contract_valid": 1.0 if not errors_for_schema and not errors else 0.0,
        "thought_leak": 1.0 if "thought_tag_leak" in errors or "thought_field_leak" in errors else 0.0,
        "markdown_fence": 1.0 if "markdown_fence" in errors else 0.0,
        "penalty": penalty,
        "schema_errors": errors_for_schema,
        "errors": errors,
        "cleaned_completion": cleaned,
        "object": obj,
    }
