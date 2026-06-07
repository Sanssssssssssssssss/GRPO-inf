from __future__ import annotations

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
    assert report["split_source_overlap"]["sources_in_multiple_splits"] >= 1
