from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


SPLITS = ("train", "dev", "test_locked")
SYSTEM_PROMPT = (
    "You are an AP evidence review agent. Output only minified valid EvidenceReviewResult JSON. "
    "Do not use markdown, prose, or chain-of-thought. Use exact source_quote strings copied from the provided documents. "
    "Do not decide payment readiness, mark invoices paid, approve payment, or submit to ERP."
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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _stats(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)

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


def compact_prompt(row: dict[str, Any]) -> str:
    input_payload = row.get("input") if isinstance(row.get("input"), dict) else {}
    gold = row.get("gold") if isinstance(row.get("gold"), dict) else {}
    payload = {
        "mode": input_payload.get("mode") or gold.get("mode"),
        "instruction": (
            "Return only valid EvidenceReviewResult JSON. Use source quotes from the provided documents. "
            "Do not decide payment readiness, mark invoices paid, approve payment, or submit to ERP."
        ),
        "user_message": input_payload.get("user_message"),
        "supervisor_task": input_payload.get("supervisor_task"),
        "case_state": input_payload.get("case_state"),
        "target_attachment_id": input_payload.get("target_attachment_id"),
        "target_evidence_id": input_payload.get("target_evidence_id"),
        "user_correction": input_payload.get("user_correction"),
        "extraction_result": input_payload.get("extraction_result"),
        "rag_context": input_payload.get("rag_context"),
        "memory_hints": input_payload.get("memory_hints"),
        "documents": row.get("documents") or [],
    }
    payload = {key: value for key, value in payload.items() if value not in (None, "", [], {})}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def chat_prompt(compact: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": compact},
    ]


def _chat_token_count(processor: Any, messages: list[dict[str, str]]) -> int:
    tokenized = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(tokenized, dict):
        tokenized = tokenized.get("input_ids", [])
    if tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return len(tokenized)


def build(source_root: Path, output_root: Path, tokenizer_path: Path, prompt_format: str = "plain", model_dir: Path | None = None) -> dict[str, Any]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    processor = None
    if prompt_format == "chat":
        if model_dir is None:
            raise ValueError("--model-dir is required when --prompt-format chat")
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)

    report: dict[str, Any] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "tokenizer_path": str(tokenizer_path),
        "prompt_format": prompt_format,
        "model_dir": str(model_dir) if model_dir else None,
        "system_prompt": SYSTEM_PROMPT if prompt_format == "chat" else None,
        "splits": {},
    }
    all_full_tokens: list[int] = []
    all_compact_tokens: list[int] = []
    all_rendered_prompt_tokens: list[int] = []
    all_full_chars: list[int] = []
    all_compact_chars: list[int] = []

    for split in SPLITS:
        src = source_root / "grpo" / f"prompts_{split}.jsonl"
        rows = _read_jsonl(src)
        out_rows: list[dict[str, Any]] = []
        full_tokens: list[int] = []
        compact_tokens: list[int] = []
        rendered_prompt_tokens: list[int] = []
        full_chars: list[int] = []
        compact_chars: list[int] = []
        for row in rows:
            compact = compact_prompt(row)
            out = dict(row)
            out["prompt_original_chars"] = len(str(row.get("prompt", "")))
            out["prompt_compact_chars"] = len(compact)
            out["prompt_format"] = prompt_format
            out["prompt_payload"] = compact
            if prompt_format == "chat":
                messages = chat_prompt(compact)
                out["prompt"] = messages
                rendered_prompt_tokens.append(_chat_token_count(processor, messages))
            else:
                out["prompt"] = compact
                rendered_prompt_tokens.append(len(tokenizer.encode(compact).ids))
            out_rows.append(out)

            original_prompt = str(row.get("prompt", ""))
            full_tokens.append(len(tokenizer.encode(original_prompt).ids))
            compact_tokens.append(len(tokenizer.encode(compact).ids))
            full_chars.append(len(original_prompt))
            compact_chars.append(len(compact))

        _write_jsonl(output_root / "grpo" / f"prompts_{split}.jsonl", out_rows)
        split_report = {
            "cases": len(rows),
            "full_prompt_tokens": _stats(full_tokens),
            "compact_prompt_tokens": _stats(compact_tokens),
            "rendered_prompt_tokens": _stats(rendered_prompt_tokens),
            "full_prompt_chars": _stats(full_chars),
            "compact_prompt_chars": _stats(compact_chars),
            "token_reduction_p50_ratio": statistics.median(compact_tokens) / statistics.median(full_tokens),
        }
        report["splits"][split] = split_report
        all_full_tokens.extend(full_tokens)
        all_compact_tokens.extend(compact_tokens)
        all_rendered_prompt_tokens.extend(rendered_prompt_tokens)
        all_full_chars.extend(full_chars)
        all_compact_chars.extend(compact_chars)

    report["overall"] = {
        "cases": len(all_full_tokens),
        "full_prompt_tokens": _stats(all_full_tokens),
        "compact_prompt_tokens": _stats(all_compact_tokens),
        "rendered_prompt_tokens": _stats(all_rendered_prompt_tokens),
        "full_prompt_chars": _stats(all_full_chars),
        "compact_prompt_chars": _stats(all_compact_chars),
        "token_reduction_p50_ratio": statistics.median(all_compact_tokens) / statistics.median(all_full_tokens),
    }
    _write_json(output_root / "compact_dataset_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build compact GRPO prompt files for CSD3 feasibility runs.")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--prompt-format", choices=("plain", "chat"), default="plain")
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.source_root, args.output_root, args.tokenizer, args.prompt_format, args.model_dir),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
