from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from grpo_inf.io import read_jsonl


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _figure_html(metrics: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return "<p>Plotly is not installed; showing summary tables only.</p>"

    parts: list[str] = []
    if metrics:
        steps = [row.get("step", index) for index, row in enumerate(metrics)]
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Reward", "EvidenceReviewResult Contract Metrics"))
        if any("reward" in row for row in metrics):
            fig.add_trace(go.Scatter(x=steps, y=[row.get("reward") for row in metrics], name="reward"), row=1, col=1)
        for key in ("json_valid_rate", "schema_valid_rate", "contract_valid_rate", "quote_hit_rate", "mode_accuracy"):
            if any(key in row for row in metrics):
                fig.add_trace(go.Scatter(x=steps, y=[row.get(key) for row in metrics], name=key), row=2, col=1)
        fig.update_layout(height=720, template="plotly_white")
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    by_category = summary.get("by_category") or {}
    if by_category:
        categories = list(by_category)
        fig = go.Figure()
        for metric in ("reward", "schema_valid_rate", "support_f1", "conflict_f1", "quote_hit_rate"):
            fig.add_bar(name=metric, x=categories, y=[by_category[cat].get(metric, 0) for cat in categories])
        fig.update_layout(barmode="group", height=520, template="plotly_white", title="Eval by scenario/category")
        parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
    return "\n".join(parts)


def _summary_table(summary: dict[str, Any]) -> str:
    rows = []
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            continue
        rows.append(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>")
    return "<table>" + "\n".join(rows) + "</table>"


def render_dashboard(run_dir: str | Path, output: str | Path | None = None) -> Path:
    run_path = Path(run_dir)
    metrics_path = run_path / "logs" / "metrics.jsonl"
    summary_path = run_path / "eval" / "summary.json"
    metrics = read_jsonl(metrics_path) if metrics_path.exists() else []
    summary = _read_json(summary_path)
    output_path = Path(output) if output else run_path / "visualizations" / "dashboard.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    worst_cases = summary.get("worst_cases") or []
    worst_html = "<ul>" + "\n".join(
        f"<li><code>{html.escape(str(row.get('case_id')))}</code> {html.escape(str(row.get('category')))} reward={html.escape(str(row.get('total')))}</li>"
        for row in worst_cases
    ) + "</ul>"

    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>GRPO Reviewer Run Dashboard</title>
  <style>
    body {{ font-family: Inter, Segoe UI, Arial, sans-serif; margin: 32px; color: #172026; }}
    table {{ border-collapse: collapse; margin: 16px 0; min-width: 520px; }}
    th, td {{ border: 1px solid #d6dde3; padding: 8px 10px; text-align: left; }}
    th {{ background: #f3f6f8; }}
    code {{ background: #eef2f5; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>GRPO Reviewer Run Dashboard</h1>
  <h2>Summary</h2>
  {_summary_table(summary)}
  <h2>Charts</h2>
  {_figure_html(metrics, summary)}
  <h2>Worst Cases</h2>
  {worst_html}
</body>
</html>
"""
    output_path.write_text(body, encoding="utf-8")
    return output_path
