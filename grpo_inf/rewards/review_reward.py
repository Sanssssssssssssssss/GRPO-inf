from __future__ import annotations

from typing import Any

from grpo_inf.rewards.context import (
    collect_review_quotes,
    context_text,
    f1,
    list_string_f1,
    normalize_scalar,
    quote_hit_rate,
    source_aliases_from_payload,
)


CLASSIFICATION_PATCH_EXEMPTIONS = {"process_only", "weak", "wrong_workflow"}


def _support_keys(result: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for item in result.get("supports") or []:
        if isinstance(item, dict):
            req = normalize_scalar(item.get("requirement"))
            level = normalize_scalar(item.get("support_level"))
            if req or level:
                keys.add(f"{req}|{level}")
    return keys


def _conflict_keys(result: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for item in result.get("conflicts") or []:
        if isinstance(item, dict):
            kind = normalize_scalar(item.get("type") or item.get("conflict_type"))
            field = normalize_scalar(item.get("field") or item.get("requirement"))
            severity = normalize_scalar(item.get("severity"))
            if kind or field or severity:
                keys.add(f"{kind}|{field}|{severity}")
    return keys


def _metadata_classification(metadata: dict[str, Any] | None, payload: dict[str, Any]) -> str:
    candidates = []
    if isinstance(metadata, dict):
        candidates.append(metadata)
    for key in ("metadata", "reward_metadata"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    for candidate in candidates:
        value = candidate.get("classification")
        if isinstance(value, dict):
            value = value.get("label") or value.get("name") or value.get("class")
        text = normalize_scalar(value)
        if text:
            return text
    return ""


def _should_penalize_unsupported_add_evidence(
    pred: dict[str, Any],
    gold: dict[str, Any],
    payload: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> bool:
    if gold.get("support_level") != "none":
        return False
    patch = pred.get("suggested_patch") if isinstance(pred.get("suggested_patch"), dict) else {}
    if not patch.get("add_evidence"):
        return False
    return _metadata_classification(metadata, payload) not in CLASSIFICATION_PATCH_EXEMPTIONS


def score_review_result(
    pred: dict[str, Any],
    gold: dict[str, Any],
    payload: dict[str, Any],
    documents: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = context_text(payload, documents)
    aliases = source_aliases_from_payload(payload)
    source_doc_id = str(pred.get("source_doc_id") or "")
    source_doc_valid = 1.0 if not aliases or source_doc_id in aliases else 0.0
    mode_ok = 1.0 if pred.get("mode") == gold.get("mode") == "review" else 0.0
    support_level_ok = 1.0 if pred.get("support_level") == gold.get("support_level") else 0.0
    should_accept_ok = 1.0 if pred.get("should_accept") == gold.get("should_accept") else 0.0
    risk_f1 = list_string_f1(pred.get("risk_flags"), gold.get("risk_flags"))
    support_f1 = f1(_support_keys(pred), _support_keys(gold))
    conflict_f1 = f1(_conflict_keys(pred), _conflict_keys(gold))
    quote_rate, quote_hit, quote_total = quote_hit_rate(collect_review_quotes(pred), text)

    pred_patch = pred.get("suggested_patch") if isinstance(pred.get("suggested_patch"), dict) else {}
    gold_patch = gold.get("suggested_patch") if isinstance(gold.get("suggested_patch"), dict) else {}
    patch_hint_ok = 1.0 if pred_patch.get("next_action_hint") == gold_patch.get("next_action_hint") else 0.0
    evidence_card_count_ok = 1.0 if bool(pred.get("evidence_cards")) == bool(gold.get("evidence_cards")) else 0.0

    component = (
        0.10 * mode_ok
        + 0.14 * support_level_ok
        + 0.12 * should_accept_ok
        + 0.16 * risk_f1
        + 0.18 * support_f1
        + 0.18 * conflict_f1
        + 0.08 * quote_rate
        + 0.02 * patch_hint_ok
        + 0.02 * evidence_card_count_ok
    )
    hallucinated_quotes = max(0, quote_total - quote_hit)
    penalty = -min(0.25, hallucinated_quotes * 0.05)
    if source_doc_valid == 0.0:
        penalty -= 0.12
    if _should_penalize_unsupported_add_evidence(pred, gold, payload, metadata):
        penalty -= 0.30

    return {
        "review_reward": max(0.0, min(1.0, component)),
        "mode_correct": mode_ok,
        "source_doc_valid": source_doc_valid,
        "support_level_correct": support_level_ok,
        "should_accept_correct": should_accept_ok,
        "risk_flag_f1": risk_f1,
        "support_f1": support_f1,
        "conflict_f1": conflict_f1,
        "quote_hit_rate": quote_rate,
        "quote_total": quote_total,
        "quote_hit": quote_hit,
        "review_patch_hint_correct": patch_hint_ok,
        "evidence_card_presence_correct": evidence_card_count_ok,
        "review_penalty": penalty,
    }
