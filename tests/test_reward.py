from __future__ import annotations

import json

from grpo_inf.rewards.reviewer_reward import score_completion


PAYLOAD = {
    "mode": "extract",
    "attachment_context": [
        {
            "attachment_id": "att_invoice_001",
            "name": "invoice_001.txt",
            "original_ref": "public/invoice_001.txt",
            "content": "INVOICE\nInvoice No: INV-100\nSupplier: Demo Medical Ltd\nTotal Due: GBP 100.00",
        }
    ],
    "extraction_context": [],
    "extraction_result": {},
}


EXTRACT_GOLD = {
    "mode": "extract",
    "source_doc_id": "invoice_001.txt",
    "evidence_type": "invoice",
    "credibility": "high",
    "extracted_fields": {
        "invoice_number": {
            "value": "INV-100",
            "status": "present",
            "source_quote": "Invoice No: INV-100",
            "source_locator": "invoice_001.txt OCR",
            "confidence": "high",
        }
    },
    "extraction_result": {"source_doc_id": "invoice_001.txt", "invoice_number": "INV-100"},
    "source_traceability": "original_document",
    "support_level": "partial",
    "risk_flags": [],
    "should_accept": False,
    "reason": "Extraction only.",
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
    "reply_to_user": "Extracted invoice fields.",
}


REVIEW_PAYLOAD = {
    "mode": "review",
    "attachment_context": PAYLOAD["attachment_context"],
    "extraction_context": [{"source_doc_id": "invoice_001.txt", "invoice_number": "INV-100", "source_quote": "Invoice No: INV-100"}],
    "extraction_result": {"source_doc_id": "invoice_001.txt", "invoice_number": "INV-100"},
}


REVIEW_GOLD = {
    **EXTRACT_GOLD,
    "mode": "review",
    "support_level": "full",
    "should_accept": True,
    "reason": "Invoice number is supported.",
    "supports": [{"requirement": "invoice_number", "support_level": "full", "quoted_text": "Invoice No: INV-100"}],
    "evidence_cards": [
        {
            "title": "Invoice INV-100",
            "doc_type": "invoice",
            "source_ref": "invoice_001.txt",
            "extracted_summary": "Invoice number INV-100.",
            "visual_summary": "OCR text only.",
            "supports": [{"requirement": "invoice_number", "support_level": "full", "quoted_text": "Invoice No: INV-100"}],
            "conflicts": [],
        }
    ],
    "suggested_patch": {
        **EXTRACT_GOLD["suggested_patch"],
        "add_evidence": [{"id": "evidence_invoice_001", "source_doc_id": "invoice_001.txt"}],
        "next_action_hint": "accept_evidence",
    },
}


def dumps(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False)


def test_extract_completion_scores_high() -> None:
    score = score_completion(dumps(EXTRACT_GOLD), EXTRACT_GOLD, payload=PAYLOAD)
    assert score["total"] > 0.95
    assert score["schema_valid"] == 1.0
    assert score["mode_correct"] == 1.0
    assert score["quote_hit_rate"] == 1.0


def test_review_completion_scores_high() -> None:
    score = score_completion(dumps(REVIEW_GOLD), REVIEW_GOLD, payload=REVIEW_PAYLOAD)
    assert score["total"] > 0.95
    assert score["support_f1"] == 1.0
    assert score["should_accept_correct"] == 1.0


def test_invalid_json_is_hard_fail() -> None:
    score = score_completion("not json", EXTRACT_GOLD, payload=PAYLOAD)
    assert score["total"] == -1.0
    assert score["json_valid"] == 0.0


def test_schema_invalid_caps_score_and_extra_fields_fail() -> None:
    bad = {**EXTRACT_GOLD, "decision": "approve"}
    score = score_completion(dumps(bad), EXTRACT_GOLD, payload=PAYLOAD)
    assert score["schema_valid"] == 0.0
    assert score["total"] <= 0.2


def test_missing_quote_penalizes_grounding() -> None:
    bad = json.loads(dumps(EXTRACT_GOLD))
    bad["extracted_fields"]["invoice_number"]["source_quote"] = "Invoice No: INV-999"
    score = score_completion(dumps(bad), EXTRACT_GOLD, payload=PAYLOAD)
    assert score["quote_hit_rate"] == 0.0
    assert score["extract_penalty"] < 0.0


def test_invalid_source_doc_id_penalized() -> None:
    bad = {**EXTRACT_GOLD, "source_doc_id": "missing.txt"}
    score = score_completion(dumps(bad), EXTRACT_GOLD, payload=PAYLOAD)
    assert score["source_doc_valid"] == 0.0
    assert score["total"] < 0.9


def test_extract_mode_forbids_add_evidence() -> None:
    bad = json.loads(dumps(EXTRACT_GOLD))
    bad["suggested_patch"]["add_evidence"] = [{"id": "not_allowed"}]
    score = score_completion(dumps(bad), EXTRACT_GOLD, payload=PAYLOAD)
    assert score["forbidden_patch_rate"] == 1.0
    assert score["extract_penalty"] < 0.0


def test_review_risk_conflict_scoring() -> None:
    gold = json.loads(dumps(REVIEW_GOLD))
    gold["support_level"] = "partial"
    gold["should_accept"] = False
    gold["risk_flags"] = ["amount_mismatch"]
    gold["supports"] = []
    gold["conflicts"] = [
        {
            "type": "amount_mismatch",
            "conflict_type": "amount_mismatch",
            "requirement": "amount_total",
            "severity": "high",
            "field": "amount_total",
            "description": "Invoice total differs from PO total.",
            "quoted_text": "Total Due: GBP 100.00",
            "conflict_with": "PO total",
            "required_follow_up": "Reconcile.",
            "affected_fields": ["amount_total"],
            "affected_evidence_ids": ["invoice_001.txt"],
            "involved_evidence_ids": ["invoice_001.txt"],
            "evidence_ids": ["invoice_001.txt"],
            "source_values": {"invoice": "GBP 100.00"},
            "suggested_resolution": "Hold.",
        }
    ]
    score = score_completion(dumps(gold), gold, payload=REVIEW_PAYLOAD)
    assert score["risk_flag_f1"] == 1.0
    assert score["conflict_f1"] == 1.0


def test_thought_and_markdown_wrappers_parse_but_penalize() -> None:
    wrapped = "<think>private reasoning</think>\n```json\n" + dumps(EXTRACT_GOLD) + "\n```"
    score = score_completion(wrapped, EXTRACT_GOLD, payload=PAYLOAD)
    assert score["json_valid"] == 1.0
    assert score["thought_leak"] == 1.0
    assert score["markdown_fence"] == 1.0
    assert score["total"] < score_completion(dumps(EXTRACT_GOLD), EXTRACT_GOLD, payload=PAYLOAD)["total"]
