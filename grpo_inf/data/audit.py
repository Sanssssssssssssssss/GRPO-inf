from __future__ import annotations

import statistics
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from grpo_inf.io import DatasetReader, SPLITS, prompt_text, write_json
from grpo_inf.rewards.context import (
    context_text,
    oracle_from_sample,
    payload_from_sample,
    source_aliases_from_payload,
    source_ids_from_source_metadata,
)
from grpo_inf.rewards.reviewer_reward import score_sample_completion
from grpo_inf.schema import schema_errors


METADATA_FIELDS_WITHOUT_QUOTES = {"source_doc_id", "source_locator", "document_confidence"}
GOLD_SCORE_TOTAL_MEAN_MIN = 0.95
GOLD_SCORE_TOTAL_MIN_MIN = 0.85
FORMAL_SOURCE_IMAGE_FIELDS = ("image", "image_path", "source_image", "source_image_path", "original_ref", "path")
FORMAL_ANNOTATION_FIELDS = (
    "annotation",
    "annotation_ref",
    "annotation_path",
    "annotation_source",
    "source_annotation",
    "public_annotation_ref",
    "label",
    "label_ref",
)


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * percentile) - 1))
    return float(ordered[index])


def _case_id(row: dict[str, Any]) -> str:
    return str(row.get("case_id") or row.get("id") or "")


def _read(reader: DatasetReader, suffix: str) -> list[dict[str, Any]]:
    return reader.read_jsonl_suffix(suffix)


def _rows_from_named_prompts(reader: DatasetReader) -> list[dict[str, Any]]:
    rows = _read(reader, "grpo/prompts.jsonl")
    if rows:
        return rows
    result: list[dict[str, Any]] = []
    for split in SPLITS:
        for row in _read(reader, f"grpo/prompts_{split}.jsonl"):
            row.setdefault("split", split)
            result.append(row)
    return result


def _rows_from_payload_answer_pairs(reader: DatasetReader) -> list[dict[str, Any]]:
    payload_rows = (
        _read(reader, "case_inputs/reviewer_payloads.jsonl")
        or _read(reader, "case_inputs/reviewer_extract_payloads.jsonl")
        or _read(reader, "system_call/reviewer_extract_payloads.jsonl")
    )
    answer_rows = _read(reader, "answers/evidence_review_result_answers.jsonl") or _read(
        reader, "answers/evidence_review_result_expected.jsonl"
    )
    if not payload_rows or not answer_rows:
        return []
    answers = {_case_id(row): row for row in answer_rows}
    rows: list[dict[str, Any]] = []
    for payload_row in payload_rows:
        cid = _case_id(payload_row)
        answer_row = answers.get(cid, {})
        payload = payload_row.get("input") if isinstance(payload_row.get("input"), dict) else {
            key: value for key, value in payload_row.items() if key not in {"case_id", "split", "scenario"}
        }
        answer = answer_row.get("answer") if isinstance(answer_row.get("answer"), dict) else {
            key: value for key, value in answer_row.items() if key not in {"case_id", "split", "scenario"}
        }
        rows.append(
            {
                "case_id": cid,
                "split": payload_row.get("split") or answer_row.get("split") or "unknown",
                "scenario": payload_row.get("scenario") or answer_row.get("scenario"),
                "input": payload,
                "gold": answer,
            }
        )
    return rows


def load_evidence_review_rows(reader: DatasetReader) -> list[dict[str, Any]]:
    rows = _rows_from_named_prompts(reader)
    if rows:
        return rows
    rows = _rows_from_payload_answer_pairs(reader)
    if rows:
        return rows
    result: list[dict[str, Any]] = []
    for split in SPLITS:
        for suffix in (f"records/{split}.jsonl", f"sft/reviewer_{split}.jsonl"):
            for row in _read(reader, suffix):
                row.setdefault("split", split)
                result.append(row)
    return result


def _quote_audit(row: dict[str, Any], strict_quotes: bool) -> tuple[int, int, list[str], list[str]]:
    payload = payload_from_sample(row)
    gold = oracle_from_sample(row)
    text = context_text(payload, row.get("documents") if isinstance(row.get("documents"), list) else [])
    total = 0
    hits = 0
    errors: list[str] = []
    warnings: list[str] = []
    fields = gold.get("extracted_fields") if isinstance(gold.get("extracted_fields"), dict) else {}
    for name, field in fields.items():
        if not isinstance(field, dict) or field.get("status") != "present":
            continue
        quote = str(field.get("source_quote") or "")
        if not quote:
            msg = f"{_case_id(row)}: present field {name} has empty source_quote"
            if strict_quotes and str(name) not in METADATA_FIELDS_WITHOUT_QUOTES:
                errors.append(msg)
            else:
                warnings.append(msg)
            continue
        total += 1
        if quote in text:
            hits += 1
        else:
            errors.append(f"{_case_id(row)}: quote not found for field {name}: {quote[:100]}")

    for item_key in ("supports", "conflicts"):
        for item in gold.get(item_key) or []:
            if not isinstance(item, dict):
                continue
            quote = str(item.get("quoted_text") or "")
            if not quote:
                continue
            total += 1
            if quote in text:
                hits += 1
            else:
                errors.append(f"{_case_id(row)}: quote not found for {item_key}: {quote[:100]}")
    return total, hits, errors, warnings


def _mode_contract_errors(row: dict[str, Any]) -> list[str]:
    gold = oracle_from_sample(row)
    errors: list[str] = []
    if gold.get("mode") == "extract":
        patch = gold.get("suggested_patch") if isinstance(gold.get("suggested_patch"), dict) else {}
        if patch.get("add_evidence"):
            errors.append(f"{_case_id(row)}: mode=extract has suggested_patch.add_evidence")
        if patch.get("evidence_items"):
            errors.append(f"{_case_id(row)}: mode=extract has suggested_patch.evidence_items")
        if gold.get("supports"):
            errors.append(f"{_case_id(row)}: mode=extract has supports")
        if gold.get("conflicts"):
            errors.append(f"{_case_id(row)}: mode=extract has conflicts")
    return errors


def _source_doc_errors(row: dict[str, Any]) -> list[str]:
    payload = payload_from_sample(row)
    gold = oracle_from_sample(row)
    aliases = source_aliases_from_payload(payload)
    source_doc_id = str(gold.get("source_doc_id") or "")
    if aliases and source_doc_id and source_doc_id not in aliases:
        return [f"{_case_id(row)}: source_doc_id not in payload attachment aliases: {source_doc_id}"]
    return []


def _source_identity(row: dict[str, Any]) -> str:
    payload = payload_from_sample(row)
    aliases = source_aliases_from_payload(payload)
    source_ids = set(aliases)
    for key in ("source", "reward_metadata"):
        value = row.get(key)
        if isinstance(value, dict):
            source_ids |= source_ids_from_source_metadata(value)
            if isinstance(value.get("source"), dict):
                source_ids |= source_ids_from_source_metadata(value["source"])
    gold = oracle_from_sample(row)
    if gold.get("source_doc_id"):
        source_ids.add(str(gold["source_doc_id"]))
    if not source_ids:
        return ""
    return sorted(source_ids)[0]


def _strict_source_identity(row: dict[str, Any]) -> tuple[str, list[str]]:
    metadata = row.get("reward_metadata")
    source = metadata.get("source") if isinstance(metadata, dict) and isinstance(metadata.get("source"), dict) else {}
    stable_source_id = str(source.get("stable_source_id") or "").strip()
    source_image_sha256 = str(source.get("source_image_sha256") or "").strip()
    source_dataset = str(source.get("source_dataset") or "").strip()
    errors: list[str] = []
    if not stable_source_id:
        errors.append(f"{_case_id(row)}: reward_metadata.source.stable_source_id is required in strict audit")
    if not source_image_sha256:
        errors.append(f"{_case_id(row)}: reward_metadata.source.source_image_sha256 is required in strict audit")
    if not source_dataset:
        errors.append(f"{_case_id(row)}: reward_metadata.source.source_dataset is required in strict audit")
    return source_image_sha256 or stable_source_id, errors


def _public_source_metadata_errors(row: dict[str, Any]) -> list[str]:
    metadata = row.get("reward_metadata")
    source = metadata.get("source") if isinstance(metadata, dict) and isinstance(metadata.get("source"), dict) else {}
    errors: list[str] = []
    if not any(str(source.get(key) or "").strip() for key in FORMAL_SOURCE_IMAGE_FIELDS):
        errors.append(f"{_case_id(row)}: public source image metadata is required")
    if not any(source.get(key) for key in FORMAL_ANNOTATION_FIELDS):
        errors.append(f"{_case_id(row)}: public annotation metadata is required")
    return errors


def _gold_self_score(rows: list[dict[str, Any]]) -> dict[str, float]:
    scores = [score_sample_completion(json.dumps(oracle_from_sample(row), ensure_ascii=False), row) for row in rows]
    totals = [float(score["total"]) for score in scores]
    schema_valid = [float(score["schema_valid"]) for score in scores]
    quote_hits = [float(score["quote_hit_rate"]) for score in scores]
    return {
        "exact_gold_total_mean": statistics.mean(totals) if totals else 1.0,
        "exact_gold_total_min": min(totals) if totals else 1.0,
        "exact_gold_schema_valid_rate": statistics.mean(schema_valid) if schema_valid else 1.0,
        "exact_gold_quote_hit_rate": statistics.mean(quote_hits) if quote_hits else 1.0,
    }


def audit_dataset(
    dataset_path: str | Path,
    output: str | Path | None = None,
    schema_name: str = "evidence_review_result",
    strict_split_source_uniqueness: bool = False,
    smoke_seed: bool = False,
    min_cases: int | None = None,
    require_extract_only: bool = False,
    require_public_source_metadata: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "schema": schema_name,
        "strict_split_source_uniqueness": strict_split_source_uniqueness,
        "require_extract_only": require_extract_only,
        "require_public_source_metadata": require_public_source_metadata,
        "smoke_seed": smoke_seed,
        "not_for_final_training": bool(smoke_seed),
        "total_cases": 0,
        "split_counts": {},
        "mode_counts": {},
        "scenario_counts": {},
        "validation_error_count": 0,
        "validation_errors_sample": [],
        "warnings": [],
    }
    if schema_name != "evidence_review_result":
        report["warnings"].append("ap_risk_ablation_schema_is_not_mainline")

    with DatasetReader(dataset_path) as reader:
        rows = load_evidence_review_rows(reader)

    report["total_cases"] = len(rows)
    gold_scores = _gold_self_score(rows)
    report.update(gold_scores)
    if gold_scores["exact_gold_total_mean"] < GOLD_SCORE_TOTAL_MEAN_MIN:
        report["validation_errors_sample"].append(
            f"exact_gold_total_mean {gold_scores['exact_gold_total_mean']:.4f} is below {GOLD_SCORE_TOTAL_MEAN_MIN}"
        )
    if gold_scores["exact_gold_total_min"] < GOLD_SCORE_TOTAL_MIN_MIN:
        report["validation_errors_sample"].append(
            f"exact_gold_total_min {gold_scores['exact_gold_total_min']:.4f} is below {GOLD_SCORE_TOTAL_MIN_MIN}"
        )
    if gold_scores["exact_gold_schema_valid_rate"] != 1.0:
        report["validation_errors_sample"].append("exact_gold_schema_valid_rate must equal 1.0")
    if gold_scores["exact_gold_quote_hit_rate"] != 1.0:
        report["validation_errors_sample"].append("exact_gold_quote_hit_rate must equal 1.0")
    if min_cases is not None and len(rows) < min_cases:
        report["validation_errors_sample"].append(f"case_count {len(rows)} is below required minimum {min_cases}")

    case_ids: list[str] = []
    prompt_lengths: list[int] = []
    quote_total = 0
    quote_hit = 0
    source_splits: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        cid = _case_id(row)
        case_ids.append(cid)
        split = str(row.get("split") or "unknown")
        report["split_counts"][split] = report["split_counts"].get(split, 0) + 1
        gold = oracle_from_sample(row)
        mode = str(gold.get("mode") or payload_from_sample(row).get("mode") or "unknown")
        report["mode_counts"][mode] = report["mode_counts"].get(mode, 0) + 1
        if require_extract_only and mode != "extract":
            report["validation_errors_sample"].append(f"{cid}: formal FATURA training rows must be mode=extract, got {mode}")
        scenario = str(row.get("scenario") or "unknown")
        report["scenario_counts"][scenario] = report["scenario_counts"].get(scenario, 0) + 1
        if prompt_text(row):
            prompt_lengths.append(len(prompt_text(row)))

        errors = schema_errors(gold)
        report["validation_errors_sample"].extend(f"{cid}: schema: {err}" for err in errors)
        report["validation_errors_sample"].extend(_mode_contract_errors(row))
        report["validation_errors_sample"].extend(_source_doc_errors(row))
        q_total, q_hit, q_errors, q_warnings = _quote_audit(row, strict_quotes=not smoke_seed)
        quote_total += q_total
        quote_hit += q_hit
        report["validation_errors_sample"].extend(q_errors)
        report["warnings"].extend(q_warnings[:10])
        if strict_split_source_uniqueness:
            source_id, strict_source_errors = _strict_source_identity(row)
            report["validation_errors_sample"].extend(strict_source_errors)
        else:
            source_id = _source_identity(row)
        if require_public_source_metadata:
            report["validation_errors_sample"].extend(_public_source_metadata_errors(row))
        if source_id:
            source_splits[source_id].add(split)
        else:
            report["warnings"].append(f"{cid}: missing_source_identity")

    duplicate_ids = len(case_ids) - len(set(case_ids))
    if duplicate_ids:
        report["validation_errors_sample"].append(f"duplicate case_id count: {duplicate_ids}")

    source_overlap = {source: sorted(splits) for source, splits in source_splits.items() if len(splits) > 1}
    report["split_source_overlap"] = {
        "sources_in_multiple_splits": len(source_overlap),
        "examples": list(source_overlap.items())[:10],
    }
    if source_overlap:
        if strict_split_source_uniqueness:
            report["validation_errors_sample"].append(
                f"source document overlap across splits: {len(source_overlap)} sources"
            )
        else:
            report["warnings"].append("source_document_overlap_across_splits")
            report["not_for_final_training"] = True

    report["quote_total"] = quote_total
    report["quote_hit"] = quote_hit
    report["quote_hit_rate"] = quote_hit / quote_total if quote_total else 1.0
    report["prompt_length_chars"] = {
        "min": min(prompt_lengths) if prompt_lengths else 0,
        "p50": statistics.median(prompt_lengths) if prompt_lengths else 0,
        "p95": _percentile(prompt_lengths, 0.95),
        "max": max(prompt_lengths) if prompt_lengths else 0,
    }
    report["mode_counts"] = dict(Counter(report["mode_counts"]))
    report["validation_errors_sample"] = report["validation_errors_sample"][:80]
    warning_set = set(report["warnings"])
    priority_warnings = [
        "source_document_overlap_across_splits",
        "ap_risk_ablation_schema_is_not_mainline",
    ]
    ordered_warnings = [warning for warning in priority_warnings if warning in warning_set]
    ordered_warnings.extend(sorted(warning_set - set(ordered_warnings))[: max(0, 80 - len(ordered_warnings))])
    report["warnings"] = ordered_warnings
    report["validation_error_count"] = len(report["validation_errors_sample"])
    report["valid"] = report["validation_error_count"] == 0

    if output:
        write_json(Path(output), report)
    return report
