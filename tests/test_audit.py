from __future__ import annotations

import json
from pathlib import Path

import pytest

from grpo_inf.data.audit import audit_dataset


PUBLIC_REVIEW_ZIP = Path("C:/Users/X/Downloads/invoice_reviewer_public_review_500_v2.zip")


def test_tiny_dataset_audit_evidence_review_result() -> None:
    report = audit_dataset("examples/tiny_dataset", smoke_seed=True)
    assert report["total_cases"] == 4
    assert report["validation_error_count"] == 0
    assert report["quote_hit_rate"] == 1.0
    assert report["mode_counts"]["extract"] == 2
    assert report["mode_counts"]["review"] == 2


def test_public_review_500_zip_is_smoke_only_when_available() -> None:
    if not PUBLIC_REVIEW_ZIP.exists():
        pytest.skip("public_review_500_v2 zip is not available")
    report = audit_dataset(PUBLIC_REVIEW_ZIP, smoke_seed=True, min_cases=500)
    assert report["total_cases"] == 500
    assert report["valid"] is True
    assert report["quote_hit_rate"] == 1.0
    assert report["not_for_final_training"] is True
    assert "source_document_overlap_across_splits" in report["warnings"]


def test_public_review_500_zip_fails_strict_split_uniqueness_when_available() -> None:
    if not PUBLIC_REVIEW_ZIP.exists():
        pytest.skip("public_review_500_v2 zip is not available")
    report = audit_dataset(PUBLIC_REVIEW_ZIP, strict_split_source_uniqueness=True, min_cases=500)
    assert report["valid"] is False
    assert any("stable_source_id is required" in error for error in report["validation_errors_sample"])


def test_strict_audit_uses_stable_source_metadata(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "grpo").mkdir(parents=True)
    rows = []
    for split, cid, source_hash in (
        ("train", "case_train", "a" * 64),
        ("dev", "case_dev", "b" * 64),
    ):
        rows.append(
            {
                "case_id": cid,
                "split": split,
                "prompt": "{\"mode\":\"extract\",\"attachment_context\":[{\"name\":\"doc.txt\",\"content\":\"Invoice No: 1\"}]}",
                "input": {"mode": "extract", "attachment_context": [{"name": "doc.txt", "content": "Invoice No: 1"}]},
                "gold": {
                    "mode": "extract",
                    "source_doc_id": "doc.txt",
                    "evidence_type": "invoice",
                    "credibility": "high",
                    "extracted_fields": {
                        "invoice_number": {
                            "value": "1",
                            "status": "present",
                            "source_quote": "Invoice No: 1",
                            "source_locator": "doc.txt OCR",
                            "confidence": "high",
                        }
                    },
                    "extraction_result": {},
                    "source_traceability": "original_document",
                    "support_level": "partial",
                    "risk_flags": [],
                    "should_accept": False,
                    "reason": "extract only",
                    "supports": [],
                    "conflicts": [],
                    "evidence_cards": [],
                    "suggested_patch": {
                        "summary": None,
                        "conversation_summary": None,
                        "case_profile": None,
                        "requirements": [],
                        "remove_requirements": [],
                        "add_evidence": [],
                        "evidence_items": [],
                        "risk_flags": [],
                        "next_questions": [],
                        "next_action_hint": "extract_current_attachment",
                        "reply_brief": None,
                        "evidence_cards": None,
                    },
                    "reply_to_user": "ok",
                },
                "reward_metadata": {
                    "source": {
                        "stable_source_id": f"fatura_{source_hash[:16]}",
                        "source_dataset": "FATURA",
                        "source_image_sha256": source_hash,
                    }
                },
            }
        )
    for split in ("train", "dev"):
        path = root / "grpo" / f"prompts_{split}.jsonl"
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows if row["split"] == split) + "\n",
            encoding="utf-8",
        )
    report = audit_dataset(root, strict_split_source_uniqueness=True, min_cases=2)
    assert report["valid"] is True
    assert report["split_source_overlap"]["sources_in_multiple_splits"] == 0
