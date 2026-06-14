from __future__ import annotations

from typing import Any

from grpo_inf.schema import REVIEWER_ANSWER_SCHEMA, parse_reviewer_json, partial_schema_score, schema_errors


STRICT_BLOCKING_ERRORS = (
    "thought_tag_leak",
    "markdown_fence",
    "json_prefix_text",
    "json_trailing_text",
    "json_recovered_first_object",
)


def _has_error_prefix(errors: list[str], prefix: str) -> bool:
    return any(error.startswith(prefix) for error in errors)


def _shape_score_from_text(text: str) -> float:
    score = 0.0
    if "{" in text:
        score += 0.04
    if "}" in text:
        score += 0.04
    required_hits = sum(1 for key in REVIEWER_ANSWER_SCHEMA["required"] if f'"{key}"' in text)
    score += min(0.12, required_hits * 0.01)
    return score


def score_system_contract(completion: Any) -> dict[str, Any]:
    obj, errors, cleaned = parse_reviewer_json(completion)
    if obj is None:
        text_shape_score = _shape_score_from_text(cleaned)
        penalty = -0.45 + text_shape_score
        if "thought_tag_leak" in errors:
            penalty -= 0.08
        if "markdown_fence" in errors:
            penalty -= 0.05
        return {
            "component": text_shape_score,
            "format_score": text_shape_score,
            "json_valid": 0.0,
            "strict_json_valid": 0.0,
            "schema_valid": 0.0,
            "contract_valid": 0.0,
            "thought_leak": 1.0 if "thought_tag_leak" in errors else 0.0,
            "gemma_thought_channel_prefix": 1.0 if "gemma_thought_channel_prefix" in errors else 0.0,
            "markdown_fence": 1.0 if "markdown_fence" in errors else 0.0,
            "wrapper_seen": 0.0,
            "trailing_extra": 1.0 if "json_trailing_text" in errors else 0.0,
            "recovered_first_json": 0.0,
            "penalty": max(-0.55, min(-0.20, penalty)),
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
    if "gemma_thought_channel_prefix" in errors:
        penalty -= 0.02
    if "markdown_fence" in errors:
        penalty -= 0.06
    if _has_error_prefix(errors, "json_wrapper:"):
        penalty -= 0.10
    if "json_trailing_text" in errors:
        penalty -= 0.10
    if "json_prefix_text" in errors:
        penalty -= 0.08
    if any(key in obj for key in ("reasoning", "chain_of_thought", "thoughts")):
        penalty -= 0.10
        errors.append("thought_field_leak")
    strict_json_valid = 0.0 if any(error in errors for error in STRICT_BLOCKING_ERRORS) else 1.0
    if _has_error_prefix(errors, "json_wrapper:"):
        strict_json_valid = 0.0
    contract_valid = 1.0 if not errors_for_schema and strict_json_valid == 1.0 else 0.0

    return {
        "component": schema_score,
        "format_score": 1.0 if strict_json_valid == 1.0 else max(0.0, min(0.5, schema_score * 0.5)),
        "json_valid": 1.0,
        "strict_json_valid": strict_json_valid,
        "schema_valid": 1.0 if not errors_for_schema else 0.0,
        "contract_valid": contract_valid,
        "thought_leak": 1.0 if "thought_tag_leak" in errors or "thought_field_leak" in errors else 0.0,
        "gemma_thought_channel_prefix": 1.0 if "gemma_thought_channel_prefix" in errors else 0.0,
        "markdown_fence": 1.0 if "markdown_fence" in errors else 0.0,
        "wrapper_seen": 1.0 if _has_error_prefix(errors, "json_wrapper:") else 0.0,
        "trailing_extra": 1.0 if "json_trailing_text" in errors else 0.0,
        "recovered_first_json": 1.0 if "json_recovered_first_object" in errors else 0.0,
        "penalty": penalty,
        "schema_errors": errors_for_schema,
        "errors": errors,
        "cleaned_completion": cleaned,
        "object": obj,
    }
