from __future__ import annotations

import math
from typing import Any

from grpo_inf.rewards.context import extract_payload_from_prompt, oracle_from_sample, parse_payload, payload_from_sample
from grpo_inf.rewards.extract_reward import score_extract_result
from grpo_inf.rewards.review_reward import score_review_result
from grpo_inf.rewards.system_contract_reward import score_system_contract


def _mode_from(gold: dict[str, Any], payload: dict[str, Any]) -> str:
    return str(gold.get("mode") or payload.get("mode") or "review")


def score_completion(
    completion: Any,
    oracle: dict[str, Any],
    documents: list[dict[str, Any]] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = score_system_contract(completion)
    obj = contract.pop("object")
    cleaned = contract.pop("cleaned_completion", "")
    if obj is None:
        return {
            "total": -1.0,
            **contract,
            "extract_reward": 0.0,
            "review_reward": 0.0,
            "active_stage_reward": 0.0,
            "mode_correct": 0.0,
            "quote_hit_rate": 0.0,
            "source_doc_valid": 0.0,
            "support_f1": 0.0,
            "conflict_f1": 0.0,
            "risk_flag_f1": 0.0,
            "forbidden_patch_rate": 0.0,
        }

    payload = payload or {}
    mode = _mode_from(oracle, payload)
    if mode == "extract":
        stage = score_extract_result(obj, oracle, payload, documents)
        stage_reward = stage["extract_reward"]
        stage_penalty = stage.get("extract_penalty", 0.0)
        review_defaults = {"review_reward": 0.0, "support_f1": 0.0, "conflict_f1": 0.0, "risk_flag_f1": 0.0}
    else:
        stage = score_review_result(obj, oracle, payload, documents)
        stage_reward = stage["review_reward"]
        stage_penalty = stage.get("review_penalty", 0.0)
        review_defaults = {}

    total = 0.35 * contract["component"] + 0.65 * stage_reward + contract["penalty"] + stage_penalty
    if contract["json_valid"] == 0.0:
        total = -1.0
    if contract["schema_valid"] == 0.0:
        total = min(total, 0.20)
    if math.isnan(total) or math.isinf(total):
        total = -1.0

    return {
        "total": max(-1.0, min(1.0, total)),
        **contract,
        **review_defaults,
        **stage,
        "active_stage_reward": stage_reward,
        "completion_chars": len(cleaned),
    }


def score_sample_completion(completion: Any, sample: dict[str, Any]) -> dict[str, Any]:
    payload = payload_from_sample(sample)
    oracle = oracle_from_sample(sample)
    documents = sample.get("documents") if isinstance(sample.get("documents"), list) else []
    return score_completion(completion, oracle, documents, payload)


def reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
    oracles = kwargs.get("gold") or kwargs.get("oracle") or kwargs.get("answer") or kwargs.get("expected_answer")
    payloads = kwargs.get("payload") or kwargs.get("input") or kwargs.get("prompt")
    documents = kwargs.get("documents")
    scores: list[float] = []
    for index, completion in enumerate(completions):
        oracle = oracles[index] if isinstance(oracles, list) and index < len(oracles) and isinstance(oracles[index], dict) else {}
        raw_payload = payloads[index] if isinstance(payloads, list) and index < len(payloads) else {}
        payload = parse_payload(raw_payload) or extract_payload_from_prompt(raw_payload)
        docs = documents[index] if isinstance(documents, list) and index < len(documents) and isinstance(documents[index], list) else []
        scores.append(score_completion(completion, oracle, docs, payload)["total"])
    return scores
