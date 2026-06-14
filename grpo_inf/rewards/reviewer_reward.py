from __future__ import annotations

import math
from typing import Any

from grpo_inf.rewards.context import extract_payload_from_prompt, oracle_from_sample, parse_payload, payload_from_sample
from grpo_inf.rewards.extract_reward import score_extract_result
from grpo_inf.rewards.review_reward import score_review_result
from grpo_inf.rewards.system_contract_reward import score_system_contract

GEMMA4_TERMINATION_TOKEN_IDS = {1, 106}
TRUNCATED_COMPLETION_CAP = 0.08
LENGTH_SOFT_CAP_TOKENS = 1200
LENGTH_HARD_CAP_TOKENS = 1536
LENGTH_MAX_PENALTY = 0.25
LOG_EXTRA_KEYS = (
    "total",
    "format_score",
    "json_valid",
    "strict_json_valid",
    "schema_valid",
    "contract_valid",
    "wrapper_seen",
    "trailing_extra",
    "recovered_first_json",
    "gemma_thought_channel_prefix",
    "thought_leak",
    "markdown_fence",
    "active_stage_reward",
    "quote_hit_rate",
    "mode_correct",
    "source_doc_valid",
    "completion_tokens",
    "eos_terminated",
    "truncated_completion",
    "length_penalty",
)


def _mode_from(gold: dict[str, Any], payload: dict[str, Any]) -> str:
    return str(gold.get("mode") or payload.get("mode") or "review")


def _completion_token_stats(completion_ids: Any) -> tuple[int, float, float]:
    if not isinstance(completion_ids, list):
        return 0, -1.0, 0.0
    token_ids = [int(token) for token in completion_ids if isinstance(token, int)]
    eos_terminated = 1.0 if any(token in GEMMA4_TERMINATION_TOKEN_IDS for token in token_ids) else 0.0
    return len(token_ids), eos_terminated, 1.0 - eos_terminated


def _length_penalty(completion_tokens: int) -> float:
    if completion_tokens <= LENGTH_SOFT_CAP_TOKENS:
        return 0.0
    over = min(completion_tokens, LENGTH_HARD_CAP_TOKENS) - LENGTH_SOFT_CAP_TOKENS
    span = LENGTH_HARD_CAP_TOKENS - LENGTH_SOFT_CAP_TOKENS
    return -min(LENGTH_MAX_PENALTY, LENGTH_MAX_PENALTY * over / span)


def score_completion(
    completion: Any,
    oracle: dict[str, Any],
    documents: list[dict[str, Any]] | None = None,
    payload: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    completion_ids: Any = None,
) -> dict[str, Any]:
    contract = score_system_contract(completion)
    obj = contract.pop("object")
    cleaned = contract.pop("cleaned_completion", "")
    completion_tokens, eos_terminated, truncated_completion = _completion_token_stats(completion_ids)
    length_penalty = _length_penalty(completion_tokens)
    if obj is None:
        total = contract["penalty"] + 0.35 * contract["component"] + length_penalty
        return {
            "total": max(-0.55, min(-0.20, total)),
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
            "completion_chars": len(cleaned),
            "completion_tokens": completion_tokens,
            "eos_terminated": eos_terminated,
            "truncated_completion": truncated_completion,
            "length_penalty": length_penalty,
        }

    payload = payload or {}
    mode = _mode_from(oracle, payload)
    if mode == "extract":
        stage = score_extract_result(obj, oracle, payload, documents)
        stage_reward = stage["extract_reward"]
        stage_penalty = stage.get("extract_penalty", 0.0)
        review_defaults = {"review_reward": 0.0, "support_f1": 0.0, "conflict_f1": 0.0, "risk_flag_f1": 0.0}
    else:
        stage = score_review_result(obj, oracle, payload, documents, metadata)
        stage_reward = stage["review_reward"]
        stage_penalty = stage.get("review_penalty", 0.0)
        review_defaults = {}

    total = 0.35 * contract["component"] + 0.65 * stage_reward + contract["penalty"] + stage_penalty
    if contract["schema_valid"] == 0.0:
        total = min(total, 0.35)
    if contract["wrapper_seen"] or contract["trailing_extra"] or contract["recovered_first_json"]:
        total = min(total, 0.15)
    total += length_penalty
    if truncated_completion == 1.0:
        total = min(total, TRUNCATED_COMPLETION_CAP)
    if math.isnan(total) or math.isinf(total):
        total = -0.55

    return {
        "total": max(-1.0, min(1.0, total)),
        **contract,
        **review_defaults,
        **stage,
        "active_stage_reward": stage_reward,
        "completion_chars": len(cleaned),
        "completion_tokens": completion_tokens,
        "eos_terminated": eos_terminated,
        "truncated_completion": truncated_completion,
        "length_penalty": length_penalty,
    }


def score_sample_completion(completion: Any, sample: dict[str, Any], completion_ids: Any = None) -> dict[str, Any]:
    payload = payload_from_sample(sample)
    oracle = oracle_from_sample(sample)
    documents = sample.get("documents") if isinstance(sample.get("documents"), list) else []
    metadata = sample.get("reward_metadata") if isinstance(sample.get("reward_metadata"), dict) else sample.get("metadata")
    return score_completion(
        completion,
        oracle,
        documents,
        payload,
        metadata if isinstance(metadata, dict) else None,
        completion_ids,
    )


def reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
    oracles = kwargs.get("gold") or kwargs.get("oracle") or kwargs.get("answer") or kwargs.get("expected_answer")
    payloads = kwargs.get("payload") or kwargs.get("input") or kwargs.get("prompt")
    documents = kwargs.get("documents")
    metadatas = kwargs.get("reward_metadata") or kwargs.get("metadata")
    completion_ids = kwargs.get("completion_ids")
    log_extra = kwargs.get("log_extra")
    log_metric = kwargs.get("log_metric")
    scores: list[float] = []
    score_rows: list[dict[str, Any]] = []
    for index, completion in enumerate(completions):
        oracle = oracles[index] if isinstance(oracles, list) and index < len(oracles) and isinstance(oracles[index], dict) else {}
        raw_payload = payloads[index] if isinstance(payloads, list) and index < len(payloads) else {}
        payload = parse_payload(raw_payload) or extract_payload_from_prompt(raw_payload)
        docs = documents[index] if isinstance(documents, list) and index < len(documents) and isinstance(documents[index], list) else []
        metadata = metadatas[index] if isinstance(metadatas, list) and index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        ids = completion_ids[index] if isinstance(completion_ids, list) and index < len(completion_ids) else None
        row = score_completion(completion, oracle, docs, payload, metadata, ids)
        score_rows.append(row)
        scores.append(row["total"])
    if callable(log_extra) and score_rows:
        for key in LOG_EXTRA_KEYS:
            log_extra(f"reward_{key}", [row.get(key) for row in score_rows])
        log_extra("reward_errors", [";".join(row.get("errors", [])) for row in score_rows])
        log_extra("reward_schema_errors", [";".join(row.get("schema_errors", [])) for row in score_rows])
    if callable(log_metric) and score_rows:
        for key in LOG_EXTRA_KEYS:
            values = [row.get(key) for row in score_rows]
            numeric = [float(value) for value in values if isinstance(value, (int, float)) and not math.isnan(float(value))]
            if numeric:
                log_metric(f"reward/{key}", sum(numeric) / len(numeric))
    return scores
