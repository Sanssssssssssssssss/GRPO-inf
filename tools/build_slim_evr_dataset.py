from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from grpo_inf.rewards.reviewer_reward import score_sample_completion
from grpo_inf.schema import schema_errors


SPLITS = ("train", "dev", "test_locked")
FIELD_KEEP_ALWAYS = (
    "invoice_number",
    "supplier",
    "buyer",
    "amount_total",
    "currency_tax",
    "po_ref",
    "grn_ref",
    "payment_terms",
)
PATCH_KEYS = (
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
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _stats(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0, "min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}

    def percentile(q: float) -> int:
        return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]

    return {
        "n": len(ordered),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _field_names_from_review(gold: dict[str, Any]) -> set[str]:
    names = set(FIELD_KEEP_ALWAYS)
    for item in gold.get("supports") or []:
        if isinstance(item, dict):
            for key in ("requirement", "field"):
                if item.get(key):
                    names.add(str(item[key]))
    for item in gold.get("conflicts") or []:
        if not isinstance(item, dict):
            continue
        for key in ("field", "requirement"):
            if item.get(key):
                names.add(str(item[key]))
        for key in ("affected_fields",):
            for value in item.get(key) or []:
                if value:
                    names.add(str(value))
    return names


def _slim_fields(gold: dict[str, Any]) -> dict[str, Any]:
    fields = gold.get("extracted_fields") if isinstance(gold.get("extracted_fields"), dict) else {}
    if gold.get("mode") == "extract":
        return fields
    keep = _field_names_from_review(gold)
    return {name: value for name, value in fields.items() if name in keep}


def _slim_support(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    return {
        "requirement": str(item.get("requirement") or item.get("field") or ""),
        "support_level": item.get("support_level") or "partial",
        "quoted_text": str(item.get("quoted_text") or item.get("source_quote") or ""),
    }


def _slim_conflict(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    result: dict[str, Any] = {}
    for key in ("type", "conflict_type", "requirement", "severity", "field", "description", "quoted_text", "conflict_with", "required_follow_up"):
        if key in item and item[key] not in (None, "", [], {}):
            result[key] = item[key]
    if "type" not in result and "conflict_type" in result:
        result["type"] = result["conflict_type"]
    if "conflict_type" not in result and "type" in result:
        result["conflict_type"] = result["type"]
    return result


def _slim_evidence_card(card: Any, source_doc_id: str, supports: list[dict[str, Any]], conflicts: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(card, dict):
        return None
    result: dict[str, Any] = {
        "source_ref": card.get("source_ref") or source_doc_id,
        "doc_type": card.get("doc_type") or "invoice",
        "title": card.get("title") or f"Review {source_doc_id}",
    }
    if supports:
        result["supports"] = supports
    if conflicts:
        result["conflicts"] = conflicts
    return result


def _slim_add_evidence(item: Any, source_doc_id: str) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    result = {}
    for key in ("id", "source_doc_id", "source_ref", "type", "evidence_type"):
        if item.get(key) not in (None, "", [], {}):
            result[key] = item[key]
    result.setdefault("source_doc_id", source_doc_id)
    return result


def _slim_patch(gold: dict[str, Any], evidence_cards: list[dict[str, Any]]) -> dict[str, Any]:
    patch = gold.get("suggested_patch") if isinstance(gold.get("suggested_patch"), dict) else {}
    source_doc_id = str(gold.get("source_doc_id") or "")
    result: dict[str, Any] = {key: None for key in PATCH_KEYS}
    result.update(
        {
            "summary": patch.get("summary"),
            "conversation_summary": None,
            "case_profile": None,
            "requirements": [],
            "remove_requirements": patch.get("remove_requirements") if isinstance(patch.get("remove_requirements"), list) else [],
            "add_evidence": [
                item
                for item in (_slim_add_evidence(value, source_doc_id) for value in patch.get("add_evidence") or [])
                if item is not None
            ],
            "evidence_items": [],
            "risk_flags": list(gold.get("risk_flags") or patch.get("risk_flags") or []),
            "next_questions": list(patch.get("next_questions") or []),
            "next_action_hint": patch.get("next_action_hint"),
            "reply_brief": None,
            "evidence_cards": evidence_cards if evidence_cards else None,
        }
    )
    return result


def _short_reason(gold: dict[str, Any]) -> str:
    mode = gold.get("mode")
    support = gold.get("support_level")
    if mode == "extract":
        return "Extraction only; no payment decision."
    if gold.get("should_accept") is True:
        return f"Evidence is {support} supported by the cited source quotes."
    return f"Evidence is {support} supported; unresolved gaps or conflicts remain."


def _short_reply(gold: dict[str, Any]) -> str:
    if gold.get("mode") == "extract":
        return "Extracted invoice evidence only; payment readiness was not reviewed."
    if gold.get("should_accept") is True:
        return "Evidence review complete; cited requirements are supported."
    return "Evidence review complete; resolve the listed gaps or conflicts before acceptance."


def slim_evr(gold: dict[str, Any]) -> dict[str, Any]:
    source_doc_id = str(gold.get("source_doc_id") or "")
    supports = [item for item in (_slim_support(value) for value in gold.get("supports") or []) if item is not None]
    conflicts = [item for item in (_slim_conflict(value) for value in gold.get("conflicts") or []) if item is not None]
    card_source = gold.get("evidence_cards") or []
    evidence_cards = [
        item
        for item in (_slim_evidence_card(value, source_doc_id, supports, conflicts) for value in card_source[:1])
        if item is not None
    ]
    patch = _slim_patch(gold, evidence_cards)
    extraction = gold.get("extraction_result") if isinstance(gold.get("extraction_result"), dict) else {}
    return {
        "mode": gold.get("mode"),
        "source_doc_id": source_doc_id,
        "evidence_type": gold.get("evidence_type"),
        "credibility": gold.get("credibility"),
        "extracted_fields": _slim_fields(gold),
        "extraction_result": {
            "source_doc_id": extraction.get("source_doc_id") or source_doc_id,
            "document_type": extraction.get("document_type") or gold.get("evidence_type"),
        },
        "source_traceability": gold.get("source_traceability"),
        "support_level": gold.get("support_level"),
        "risk_flags": list(gold.get("risk_flags") or []),
        "should_accept": bool(gold.get("should_accept")),
        "reason": _short_reason(gold),
        "supports": supports,
        "conflicts": conflicts,
        "evidence_cards": evidence_cards,
        "suggested_patch": patch,
        "reply_to_user": _short_reply(gold),
    }


def _answer_text(gold: dict[str, Any]) -> str:
    return json.dumps(gold, ensure_ascii=False, separators=(",", ":"))


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build(source_root: Path, output_root: Path, model_dir: Path | None = None) -> dict[str, Any]:
    tokenizer = None
    if model_dir is not None:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    report: dict[str, Any] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "model_dir": str(model_dir) if model_dir else None,
        "slim_gold_version": "slim_evr_v1",
        "splits": {},
    }
    all_original_chars: list[int] = []
    all_slim_chars: list[int] = []
    all_original_tokens: list[int] = []
    all_slim_tokens: list[int] = []
    all_scores: list[float] = []
    all_schema_valid: list[float] = []
    all_quote_hits: list[float] = []

    for split in SPLITS:
        source_path = source_root / "grpo" / f"prompts_{split}.jsonl"
        if not source_path.exists():
            continue
        rows = _read_jsonl(source_path)
        out_rows: list[dict[str, Any]] = []
        original_chars: list[int] = []
        slim_chars: list[int] = []
        original_tokens: list[int] = []
        slim_tokens: list[int] = []
        scores: list[float] = []
        schema_valid: list[float] = []
        quote_hits: list[float] = []
        for row in rows:
            gold = row.get("gold")
            if not isinstance(gold, dict):
                raise ValueError(f"{row.get('case_id', '<unknown>')} is missing dict gold")
            slim = slim_evr(gold)
            errors = schema_errors(slim)
            if errors:
                raise ValueError(f"{row.get('case_id', '<unknown>')} has schema-invalid slim gold: {errors[:3]}")
            out = dict(row)
            out["gold"] = slim
            out["gold_slim_version"] = "slim_evr_v1"
            out_rows.append(out)

            original_text = _answer_text(gold)
            slim_text = _answer_text(slim)
            original_chars.append(len(original_text))
            slim_chars.append(len(slim_text))
            if tokenizer is not None:
                original_tokens.append(_token_count(tokenizer, original_text + "<turn|>\n"))
                slim_tokens.append(_token_count(tokenizer, slim_text + "<turn|>\n"))
            score = score_sample_completion(slim_text, out)
            scores.append(float(score["total"]))
            schema_valid.append(float(score["schema_valid"]))
            quote_hits.append(float(score["quote_hit_rate"]))
        _write_jsonl(output_root / "grpo" / f"prompts_{split}.jsonl", out_rows)
        split_report: dict[str, Any] = {
            "cases": len(rows),
            "original_completion_chars": _stats(original_chars),
            "slim_completion_chars": _stats(slim_chars),
            "char_reduction_p50_ratio": statistics.median(slim_chars) / statistics.median(original_chars),
            "exact_gold_total_mean": sum(scores) / len(scores),
            "exact_gold_total_min": min(scores),
            "exact_gold_schema_valid_rate": sum(schema_valid) / len(schema_valid),
            "exact_gold_quote_hit_rate": sum(quote_hits) / len(quote_hits),
        }
        if tokenizer is not None:
            split_report["original_completion_tokens"] = _stats(original_tokens)
            split_report["slim_completion_tokens"] = _stats(slim_tokens)
            split_report["token_reduction_p50_ratio"] = statistics.median(slim_tokens) / statistics.median(original_tokens)
        report["splits"][split] = split_report
        all_original_chars.extend(original_chars)
        all_slim_chars.extend(slim_chars)
        all_original_tokens.extend(original_tokens)
        all_slim_tokens.extend(slim_tokens)
        all_scores.extend(scores)
        all_schema_valid.extend(schema_valid)
        all_quote_hits.extend(quote_hits)

    report["overall"] = {
        "cases": len(all_slim_chars),
        "original_completion_chars": _stats(all_original_chars),
        "slim_completion_chars": _stats(all_slim_chars),
        "char_reduction_p50_ratio": statistics.median(all_slim_chars) / statistics.median(all_original_chars),
        "exact_gold_total_mean": sum(all_scores) / len(all_scores),
        "exact_gold_total_min": min(all_scores),
        "exact_gold_schema_valid_rate": sum(all_schema_valid) / len(all_schema_valid),
        "exact_gold_quote_hit_rate": sum(all_quote_hits) / len(all_quote_hits),
    }
    if tokenizer is not None:
        report["overall"]["original_completion_tokens"] = _stats(all_original_tokens)
        report["overall"]["slim_completion_tokens"] = _stats(all_slim_tokens)
        report["overall"]["token_reduction_p50_ratio"] = statistics.median(all_slim_tokens) / statistics.median(all_original_tokens)
    _write_json(output_root / "slim_evr_dataset_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build schema-compatible slim EvidenceReviewResult gold targets.")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source_root, args.output_root, args.model_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
