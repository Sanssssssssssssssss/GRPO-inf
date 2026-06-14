from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoProcessor
from trl.trainer.utils import create_model_from_path

from grpo_inf.rewards.reviewer_reward import GEMMA4_TERMINATION_TOKEN_IDS, score_sample_completion


def _read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _prompt_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        return prompt
    raise ValueError(f"{row.get('case_id', '<unknown>')} is missing conversational prompt")


def _generate_one(model: Any, processor: Any, row: dict[str, Any], max_new_tokens: int) -> tuple[str, list[int]]:
    inputs = processor.apply_chat_template(
        _prompt_from_row(row),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=sorted(GEMMA4_TERMINATION_TOKEN_IDS),
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    completion_ids = generated[0, prompt_len:].detach().cpu().tolist()
    text = processor.tokenizer.decode(completion_ids, skip_special_tokens=True)
    return text, completion_ids


def audit(
    model_dir: Path,
    adapter_dir: Path,
    data_path: Path,
    output_jsonl: Path,
    summary_out: Path,
    limit: int,
    max_new_tokens: int,
    min_schema_valid_rate: float,
    min_contract_valid_rate: float,
    max_clipped_rate: float,
) -> dict[str, Any]:
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, truncation_side="left", padding_side="left")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    base = create_model_from_path(
        str(model_dir),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    model.eval()

    rows = _read_jsonl(data_path, limit)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    scored_rows: list[dict[str, Any]] = []
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            completion, completion_ids = _generate_one(model, processor, row, max_new_tokens)
            score = score_sample_completion(completion, row, completion_ids)
            eos_terminated = any(token in GEMMA4_TERMINATION_TOKEN_IDS for token in completion_ids)
            payload = {
                "case_id": row.get("case_id"),
                "completion": completion,
                "completion_tokens": len(completion_ids),
                "eos_terminated": eos_terminated,
                "score": score,
            }
            scored_rows.append(payload)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()

    n = len(scored_rows) or 1
    summary = {
        "model_dir": str(model_dir),
        "adapter_dir": str(adapter_dir),
        "data_path": str(data_path),
        "cases": len(scored_rows),
        "mean_reward": sum(float(row["score"]["total"]) for row in scored_rows) / n,
        "schema_valid_rate": sum(1 for row in scored_rows if row["score"].get("schema_valid") == 1.0) / n,
        "contract_valid_rate": sum(1 for row in scored_rows if row["score"].get("contract_valid") == 1.0) / n,
        "quote_hit_rate_mean": sum(float(row["score"].get("quote_hit_rate", 0.0)) for row in scored_rows) / n,
        "eos_terminated_rate": sum(1 for row in scored_rows if row["eos_terminated"]) / n,
        "clipped_rate": sum(1 for row in scored_rows if not row["eos_terminated"]) / n,
        "completion_tokens_max": max((row["completion_tokens"] for row in scored_rows), default=0),
        "max_new_tokens": max_new_tokens,
        "output_jsonl": str(output_jsonl),
    }
    summary["thresholds"] = {
        "min_schema_valid_rate": min_schema_valid_rate,
        "min_contract_valid_rate": min_contract_valid_rate,
        "max_clipped_rate": max_clipped_rate,
    }
    failures: list[str] = []
    if summary["schema_valid_rate"] < min_schema_valid_rate:
        failures.append(f"schema_valid_rate {summary['schema_valid_rate']:.4f} < {min_schema_valid_rate:.4f}")
    if summary["contract_valid_rate"] < min_contract_valid_rate:
        failures.append(f"contract_valid_rate {summary['contract_valid_rate']:.4f} < {min_contract_valid_rate:.4f}")
    if summary["clipped_rate"] > max_clipped_rate:
        failures.append(f"clipped_rate {summary['clipped_rate']:.4f} > {max_clipped_rate:.4f}")
    summary["valid"] = not failures
    summary["failures"] = failures
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and score dev cases for a Gemma4 LoRA adapter.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--min-schema-valid-rate", type=float, default=0.0)
    parser.add_argument("--min-contract-valid-rate", type=float, default=0.0)
    parser.add_argument("--max-clipped-rate", type=float, default=1.0)
    args = parser.parse_args()
    summary = audit(
        args.model_dir,
        args.adapter_dir,
        args.data_path,
        args.output_jsonl,
        args.summary_out,
        args.limit,
        args.max_new_tokens,
        args.min_schema_valid_rate,
        args.min_contract_valid_rate,
        args.max_clipped_rate,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
