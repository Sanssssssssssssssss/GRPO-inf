from __future__ import annotations

from pathlib import Path
import os

import pytest

from grpo_inf.data.audit import audit_dataset


def test_tiny_dataset_audit_detects_overlap() -> None:
    report = audit_dataset("examples/tiny_dataset")
    assert report["total_cases"] == 4
    assert report["validation_error_count"] == 0
    assert report["quote_hit_rate"] == 1.0
    assert "vendor_overlap_across_splits" in report["warnings"]
    assert "template_overlap_across_splits" in report["warnings"]


def test_attached_zip_audit_when_available() -> None:
    zip_env = os.environ.get("INVOICE_REVIEWER_DATASET_ZIP")
    if not zip_env:
        pytest.skip("set INVOICE_REVIEWER_DATASET_ZIP to audit the attached seed zip")
    zip_path = Path(zip_env)
    if not zip_path.exists():
        pytest.skip("attached seed dataset zip is not available")
    report = audit_dataset(zip_path)
    assert report["total_cases"] == 1650
    assert report["quote_hit_rate"] == 1.0
    assert "vendor_overlap_across_splits" in report["warnings"]
    assert report["split_overlap"]["vendors_in_multiple_splits"] >= 1
