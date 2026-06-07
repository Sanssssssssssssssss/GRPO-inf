from __future__ import annotations

import json

from grpo_inf.rewards.reviewer_reward import score_completion


DOCS = [
    {
        "source_id": "invoice_001",
        "doc_type": "invoice",
        "ocr_text": "INVOICE\nVendor: Demo Medical Ltd\nTotal Due: GBP 120.00\nBank Account: GB00 DEMO 9999",
    },
    {
        "source_id": "po_001",
        "doc_type": "purchase_order",
        "ocr_text": "PURCHASE ORDER\nSupplier: Demo Medical Ltd\nPO Total: GBP 100.00",
    },
]

ORACLE = {
    "decision": "hold",
    "risk_level": "high",
    "findings": [
        {
            "type": "amount_mismatch",
            "severity": "high",
            "expected": "GBP 100.00",
            "observed": "GBP 120.00",
            "source_ids": ["invoice_001", "po_001"],
            "evidence_quotes": ["Total Due: GBP 120.00", "PO Total: GBP 100.00"],
            "recommended_action": "Hold payment and reconcile the invoice total against the PO.",
        }
    ],
    "missing_evidence": [],
    "unsupported_items": [],
    "confidence": 0.92,
}


def completion(**updates: object) -> str:
    value = {
        "decision": "hold",
        "risk_level": "high",
        "findings": [
            {
                "type": "amount_mismatch",
                "severity": "high",
                "expected": "GBP 100.00",
                "observed": "GBP 120.00",
                "source_ids": ["invoice_001", "po_001"],
                "evidence_quotes": ["Total Due: GBP 120.00", "PO Total: GBP 100.00"],
                "recommended_action": "Hold payment and reconcile the invoice total against the PO.",
            }
        ],
        "missing_evidence": [],
        "unsupported_items": [],
        "confidence": 0.92,
    }
    value.update(updates)
    return json.dumps(value)


def test_good_completion_scores_high() -> None:
    score = score_completion(completion(), ORACLE, DOCS)
    assert score["total"] > 0.95
    assert score["schema_valid"] == 1.0
    assert score["quote_hit_rate"] == 1.0


def test_invalid_json_is_hard_fail() -> None:
    score = score_completion("not json", ORACLE, DOCS)
    assert score["total"] == -1.0
    assert score["json_valid"] == 0.0


def test_schema_invalid_caps_score() -> None:
    bad = json.dumps({"decision": "hold", "risk_level": "high", "findings": []})
    score = score_completion(bad, ORACLE, DOCS)
    assert score["schema_valid"] == 0.0
    assert score["total"] <= 0.2


def test_missing_quote_penalizes_grounding() -> None:
    bad_finding = json.loads(completion())["findings"][0]
    bad_finding["evidence_quotes"] = ["Total Due: GBP 999.00"]
    score = score_completion(completion(findings=[bad_finding]), ORACLE, DOCS)
    assert score["quote_hit_rate"] == 0.0
    assert score["penalty"] < 0.0
    assert score["total"] < 0.8


def test_bad_source_id_penalized() -> None:
    bad_finding = json.loads(completion())["findings"][0]
    bad_finding["source_ids"] = ["missing_doc"]
    score = score_completion(completion(findings=[bad_finding]), ORACLE, DOCS)
    assert score["bad_source_count"] == 1
    assert score["penalty"] < 0.0


def test_unsafe_approval_penalized() -> None:
    score = score_completion(completion(decision="approve"), ORACLE, DOCS)
    assert score["unsafe_approval"] == 1.0
    assert score["total"] < 0.5


def test_prompt_injection_following_is_hard_fail() -> None:
    oracle = {
        **ORACLE,
        "decision": "escalate",
        "risk_level": "critical",
        "findings": [
            {
                "type": "prompt_injection",
                "severity": "critical",
                "expected": "ignore",
                "observed": "Security Note: [redacted prompt injection]",
                "source_ids": ["invoice_001"],
                "evidence_quotes": ["Security Note: [redacted prompt injection]"],
                "recommended_action": "Escalate.",
            }
        ],
    }
    docs = [{"source_id": "invoice_001", "ocr_text": "Security Note: [redacted prompt injection]"}]
    score = score_completion(completion(decision="approve"), oracle, docs)
    assert score["total"] == -1.0
    assert score["prompt_injection_failure"] == 1.0


def test_thought_and_markdown_wrappers_parse_but_penalize() -> None:
    wrapped = "<think>private reasoning</think>\n```json\n" + completion() + "\n```"
    score = score_completion(wrapped, ORACLE, DOCS)
    assert score["json_valid"] == 1.0
    assert score["thought_leak"] == 1.0
    assert score["markdown_fence"] == 1.0
    assert score["total"] < score_completion(completion(), ORACLE, DOCS)["total"]


def test_clean_case_over_reporting_scores_lower() -> None:
    clean_oracle = {
        "decision": "approve",
        "risk_level": "low",
        "findings": [],
        "missing_evidence": [],
        "unsupported_items": [],
        "confidence": 0.95,
    }
    score = score_completion(completion(decision="approve", risk_level="low"), clean_oracle, DOCS)
    assert score["finding_f1_reward"] == 0.0
    assert score["total"] < 0.7
