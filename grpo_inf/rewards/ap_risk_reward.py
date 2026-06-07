from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from grpo_inf.schema.ap_risk_ablation import AP_RISK_ABLATION_SCHEMA
from grpo_inf.schema.evidence_review_result import parse_reviewer_json


AP_RISK_VALIDATOR = Draft202012Validator(AP_RISK_ABLATION_SCHEMA)


def score_ap_risk_completion(completion: Any, oracle: dict[str, Any], documents: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    obj, errors, _ = parse_reviewer_json(completion)
    if obj is None:
        return {"total": -1.0, "json_valid": 0.0, "schema_valid": 0.0, "errors": errors}
    schema_errors = [err.message for err in AP_RISK_VALIDATOR.iter_errors(obj)]
    if schema_errors:
        return {"total": 0.0, "json_valid": 1.0, "schema_valid": 0.0, "schema_errors": schema_errors, "errors": errors}
    decision_ok = obj.get("decision") == oracle.get("decision")
    risk_ok = obj.get("risk_level") == oracle.get("risk_level")
    pred_types = {str(item.get("type")) for item in obj.get("findings", []) if isinstance(item, dict)}
    gold_types = {str(item.get("type")) for item in oracle.get("findings", []) if isinstance(item, dict)}
    finding_ok = pred_types == gold_types
    total = (0.35 if decision_ok else 0.0) + (0.25 if risk_ok else 0.0) + (0.40 if finding_ok else 0.0)
    return {"total": total, "json_valid": 1.0, "schema_valid": 1.0, "errors": errors}
