from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def stable_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_scalar(value: Any) -> str:
    return " ".join(stable_text(value).strip().lower().split())


def f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def parse_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def payload_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    for key in ("payload", "input", "prompt_payload"):
        payload = parse_payload(sample.get(key))
        if payload:
            return payload
    prompt = sample.get("prompt")
    if isinstance(prompt, str):
        return parse_payload(prompt)
    if isinstance(prompt, list):
        for item in reversed(prompt):
            if isinstance(item, dict) and item.get("role") == "user":
                payload = parse_payload(item.get("content"))
                if payload:
                    return payload
    return {}


def oracle_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    for key in ("gold", "answer", "expected_answer", "expected_output", "oracle", "reviewer_oracle"):
        value = sample.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _basename(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return os.path.basename(text.replace("\\", "/"))


def source_aliases_from_payload(payload: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for item in payload.get("attachment_context") or []:
        if not isinstance(item, dict):
            continue
        for key in ("source_doc_id", "attachment_id", "name", "path", "source_path", "original_ref"):
            value = str(item.get(key) or "").strip()
            if value:
                aliases.add(value)
                base = _basename(value)
                if base:
                    aliases.add(base)
    extraction = payload.get("extraction_result")
    if isinstance(extraction, dict):
        for key in ("source_doc_id", "attachment_id", "document_id"):
            value = str(extraction.get(key) or "").strip()
            if value:
                aliases.add(value)
                base = _basename(value)
                if base:
                    aliases.add(base)
    return aliases


def source_ids_from_source_metadata(source: Any) -> set[str]:
    if not isinstance(source, dict):
        return set()
    ids: set[str] = set()
    for key in ("source_id", "source_doc_id", "source_image_id", "image", "original_ref", "path"):
        value = str(source.get(key) or "").strip()
        if value:
            ids.add(value)
            base = _basename(value)
            if base:
                ids.add(base)
    return ids


def context_text(payload: dict[str, Any], documents: list[dict[str, Any]] | None = None) -> str:
    parts: list[str] = []
    for item in payload.get("attachment_context") or []:
        if isinstance(item, dict):
            for key in ("content", "text", "ocr_text", "raw_text"):
                if item.get(key):
                    parts.append(stable_text(item[key]))
    for item in payload.get("extraction_context") or []:
        if isinstance(item, dict):
            parts.append(stable_text(item))
        elif item:
            parts.append(stable_text(item))
    extraction = payload.get("extraction_result")
    if extraction:
        parts.append(stable_text(extraction))
    for doc in documents or []:
        if isinstance(doc, dict):
            for key in ("ocr_text", "text", "raw_text", "content"):
                if doc.get(key):
                    parts.append(stable_text(doc[key]))
    return "\n".join(parts)


def quote_hit_rate(quotes: list[str], text: str) -> tuple[float, int, int]:
    nonempty = [quote for quote in quotes if str(quote).strip()]
    if not nonempty:
        return 1.0, 0, 0
    hits = sum(1 for quote in nonempty if str(quote) in text)
    return hits / len(nonempty), hits, len(nonempty)


def collect_field_quotes(result: dict[str, Any]) -> list[str]:
    quotes: list[str] = []
    fields = result.get("extracted_fields")
    if isinstance(fields, dict):
        for field in fields.values():
            if isinstance(field, dict) and field.get("source_quote"):
                quotes.append(str(field["source_quote"]))
    extraction = result.get("extraction_result")
    if isinstance(extraction, dict):
        for item in extraction.get("field_inventory") or []:
            if isinstance(item, dict) and item.get("source_quote"):
                quotes.append(str(item["source_quote"]))
    return quotes


def collect_review_quotes(result: dict[str, Any]) -> list[str]:
    quotes: list[str] = []
    for key in ("supports", "conflicts"):
        for item in result.get(key) or []:
            if isinstance(item, dict):
                for quote_key in ("quoted_text", "source_quote"):
                    if item.get(quote_key):
                        quotes.append(str(item[quote_key]))
    for card in result.get("evidence_cards") or []:
        if not isinstance(card, dict):
            continue
        for key in ("supports", "conflicts"):
            for item in card.get(key) or []:
                if isinstance(item, dict) and item.get("quoted_text"):
                    quotes.append(str(item["quoted_text"]))
    return quotes


def field_names_by_status(result: dict[str, Any], statuses: set[str] | None = None) -> set[str]:
    fields = result.get("extracted_fields")
    if not isinstance(fields, dict):
        return set()
    names: set[str] = set()
    for name, field in fields.items():
        if not isinstance(field, dict):
            continue
        if statuses is None or str(field.get("status")) in statuses:
            names.add(str(name))
    return names


def field_value_score(pred: dict[str, Any], gold: dict[str, Any]) -> float:
    pred_fields = pred.get("extracted_fields") if isinstance(pred.get("extracted_fields"), dict) else {}
    gold_fields = gold.get("extracted_fields") if isinstance(gold.get("extracted_fields"), dict) else {}
    if not gold_fields:
        return 1.0 if not pred_fields else 0.0
    scores: list[float] = []
    for name, gold_field in gold_fields.items():
        pred_field = pred_fields.get(name)
        if not isinstance(gold_field, dict) or not isinstance(pred_field, dict):
            scores.append(0.0)
            continue
        status_ok = pred_field.get("status") == gold_field.get("status")
        gold_value = normalize_scalar(gold_field.get("value"))
        pred_value = normalize_scalar(pred_field.get("value"))
        if not gold_value and not pred_value:
            value_ok = True
        else:
            value_ok = bool(gold_value and pred_value and (gold_value == pred_value or gold_value in pred_value or pred_value in gold_value))
        scores.append((0.35 if status_ok else 0.0) + (0.65 if value_ok else 0.0))
    return sum(scores) / len(scores)


def list_string_f1(pred_values: Any, gold_values: Any) -> float:
    pred = {normalize_scalar(value) for value in pred_values or [] if normalize_scalar(value)}
    gold = {normalize_scalar(value) for value in gold_values or [] if normalize_scalar(value)}
    return f1(pred, gold)
