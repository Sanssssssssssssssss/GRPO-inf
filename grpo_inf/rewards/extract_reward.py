from __future__ import annotations

from typing import Any

from grpo_inf.rewards.context import (
    collect_field_quotes,
    context_text,
    f1,
    field_names_by_status,
    field_value_score,
    quote_hit_rate,
    source_aliases_from_payload,
)


def score_extract_result(pred: dict[str, Any], gold: dict[str, Any], payload: dict[str, Any], documents: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    text = context_text(payload, documents)
    source_aliases = source_aliases_from_payload(payload)
    source_doc_id = str(pred.get("source_doc_id") or "")

    mode_ok = 1.0 if pred.get("mode") == "extract" else 0.0
    source_doc_valid = 1.0 if not source_aliases or source_doc_id in source_aliases else 0.0
    value_score = field_value_score(pred, gold)
    present_f1 = f1(
        field_names_by_status(pred, {"present"}),
        field_names_by_status(gold, {"present"}),
    )
    missing_f1 = f1(
        field_names_by_status(pred, {"missing", "unclear", "conflict"}),
        field_names_by_status(gold, {"missing", "unclear", "conflict"}),
    )
    quote_rate, quote_hit, quote_total = quote_hit_rate(collect_field_quotes(pred), text)

    suggested_patch = pred.get("suggested_patch") if isinstance(pred.get("suggested_patch"), dict) else {}
    add_evidence = suggested_patch.get("add_evidence") or []
    evidence_items = suggested_patch.get("evidence_items") or []
    extract_patch_ok = 1.0 if not add_evidence and not evidence_items and not pred.get("supports") and not pred.get("conflicts") else 0.0
    should_accept_ok = 1.0 if pred.get("should_accept") is False else 0.0

    component = (
        0.12 * mode_ok
        + 0.12 * source_doc_valid
        + 0.28 * value_score
        + 0.14 * present_f1
        + 0.08 * missing_f1
        + 0.16 * quote_rate
        + 0.06 * extract_patch_ok
        + 0.04 * should_accept_ok
    )
    hallucinated_quotes = max(0, quote_total - quote_hit)
    penalty = -min(0.30, hallucinated_quotes * 0.05)
    if source_doc_valid == 0.0:
        penalty -= 0.12
    if pred.get("mode") == "extract" and add_evidence:
        penalty -= 0.20
    lowered = str(pred).lower()
    if "approve payment" in lowered or "付款" in lowered or "mark as paid" in lowered:
        penalty -= 0.15

    return {
        "extract_reward": max(0.0, min(1.0, component)),
        "mode_correct": mode_ok,
        "source_doc_valid": source_doc_valid,
        "extract_field_value_score": value_score,
        "extract_present_field_f1": present_f1,
        "extract_missing_field_f1": missing_f1,
        "quote_hit_rate": quote_rate,
        "quote_total": quote_total,
        "quote_hit": quote_hit,
        "forbidden_patch_rate": 0.0 if extract_patch_ok else 1.0,
        "extract_should_accept_valid": should_accept_ok,
        "extract_penalty": penalty,
    }
