import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import jinja2
from app.events.stream import TraceView
from app.data.ingestion import ImageMetadataEnvelope
from app.storage.sandbox import StorageSandbox

REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SatQuery AI — Execution Audit Report</title>
    <style>
        :root {
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border: #30363d;
            --accent: #58a6ff;
            --cyan: #39c5bb;
            --text: #c9d1d9;
            --text-heading: #f0f6fc;
            --badge-bg: #21262d;
            --success: #3fb950;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text);
            margin: 0;
            padding: 40px;
            line-height: 1.6;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        .header {
            border-bottom: 2px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        .title {
            color: var(--text-heading);
            font-size: 28px;
            font-weight: 700;
            margin: 0 0 6px 0;
        }
        .subtitle {
            color: var(--accent);
            font-size: 14px;
            font-weight: 500;
        }
        .meta-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 30px;
        }
        .meta-item strong {
            display: block;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #8b949e;
        }
        .meta-item span {
            font-size: 14px;
            color: var(--text-heading);
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 25px;
        }
        .card h2 {
            color: var(--text-heading);
            font-size: 18px;
            margin-top: 0;
            margin-bottom: 16px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .badge {
            background: var(--badge-bg);
            border: 1px solid var(--border);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            color: var(--cyan);
        }
        .answer-box {
            font-size: 15px;
            white-space: pre-wrap;
            line-height: 1.7;
            color: var(--text-heading);
        }
        .evidence-img {
            max-width: 100%;
            border-radius: 6px;
            border: 1px solid var(--border);
            display: block;
            margin-top: 15px;
        }
        .inputs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }
        .input-card {
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 16px;
        }
        .input-card img {
            width: 100%;
            height: 180px;
            object-fit: cover;
            border-radius: 4px;
            margin-bottom: 12px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        th, td {
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
        }
        th {
            color: #8b949e;
            font-weight: 600;
        }
        .status-pill {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }
        .status-SUCCESS { background: rgba(63, 185, 80, 0.2); color: #3fb950; }
        .status-INFO { background: rgba(88, 166, 255, 0.2); color: #58a6ff; }
        .footer {
            text-align: center;
            font-size: 12px;
            color: #8b949e;
            margin-top: 40px;
            border-top: 1px solid var(--border);
            padding-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1 class="title">SatQuery AI — Analysis Report</h1>
                <div class="subtitle">SIH 2026 Problem Statement 26167 | ISRO / Department of Space</div>
            </div>
            <div class="badge">{{ trace_view.status }}</div>
        </div>

        <div class="meta-grid">
            <div class="meta-item">
                <strong>Trace ID</strong>
                <span>{{ trace_view.trace_id }}</span>
            </div>
            <div class="meta-item">
                <strong>Task Type</strong>
                <span>{{ trace_view.task or 'N/A' }}</span>
            </div>
            <div class="meta-item">
                <strong>Specialist Tool</strong>
                <span>{{ trace_view.selected_tool or 'N/A' }}</span>
            </div>
            <div class="meta-item">
                <strong>Confidence</strong>
                <span style="color: var(--cyan); font-weight: bold;">
                    {% if trace_view.confidence %}{{ (trace_view.confidence * 100) | round(1) }}%{% else %}N/A{% endif %}
                </span>
            </div>
        </div>

        <div class="card">
            <h2>Natural Language Query</h2>
            <div style="font-size: 17px; font-weight: 500; color: #fff;">"{{ query }}"</div>
        </div>

        <div class="card">
            <h2>Input Imagery Specifications</h2>
            <div class="inputs-grid">
                {% for img in images %}
                <div class="input-card">
                    {% if img.thumbnail_base64 %}
                    <img src="{{ img.thumbnail_base64 }}" alt="{{ img.filename }}">
                    {% endif %}
                    <div style="font-weight: 600; color: #f0f6fc; margin-bottom: 6px;">{{ img.filename }}</div>
                    <div style="font-size: 12px; color: #8b949e;">
                        Modality: <span class="badge" style="font-size: 11px;">{{ img.modality | upper }}</span><br>
                        Dimensions: {{ img.width }} &times; {{ img.height }} px ({{ img.bands }} bands)<br>
                        CRS: {{ img.crs or 'Local coordinates' }}<br>
                        SHA-256: <code>{{ img.sha256[:16] }}...</code>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>

        <div class="card">
            <h2>
                <span>Evidence-Grounded Response</span>
                <span class="badge" style="font-size: 11px;">{{ trace_view.selected_model or 'Specialist' }}</span>
            </h2>
            <div class="answer-box">{{ answer }}</div>

            {% if evidence_url %}
            <h3 style="color: var(--text-heading); font-size: 14px; margin-top: 25px; margin-bottom: 8px;">Spatial Evidence Overlay</h3>
            <img class="evidence-img" src="{{ evidence_url }}" alt="Evidence Overlay">
            {% endif %}
        </div>

        <div class="card">
            <h2>Auditable Agentic Execution Trace</h2>
            <table>
                <thead>
                    <tr>
                        <th>Step</th>
                        <th>Status</th>
                        <th>Component</th>
                        <th>Duration</th>
                        <th>Timestamp</th>
                    </tr>
                </thead>
                <tbody>
                    {% for step in trace_view.steps %}
                    <tr>
                        <td style="font-weight: 500; color: #f0f6fc;">{{ step.step_name }}</td>
                        <td><span class="status-pill status-{{ step.status }}">{{ step.status }}</span></td>
                        <td>{{ step.tool_name or step.model_name or 'Controller' }}</td>
                        <td>{% if step.duration_ms %}{{ step.duration_ms }} ms{% else %}—{% endif %}</td>
                        <td style="color: #8b949e;">{{ step.timestamp[11:19] }} UTC</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="footer">
            Generated autonomously by SatQuery AI Agentic Backend • Deterministic Greedy Inference • Total Latency: {{ trace_view.total_duration_ms }} ms
        </div>
    </div>
</body>
</html>
"""


class ReportGenerator:
    """Generates auditable, self-contained HTML reports for remote sensing queries."""

    def __init__(self, sandbox: StorageSandbox):
        self.sandbox = sandbox
        self.jinja_env = jinja2.Environment(loader=jinja2.BaseLoader())
        self.template = self.jinja_env.from_string(REPORT_TEMPLATE)

    def generate_html_report(
        self,
        session_id: str,
        query: str,
        answer: str,
        images: List[ImageMetadataEnvelope],
        trace_view: TraceView,
        evidence_path: Optional[Path] = None,
        report_id: Optional[str] = None
    ) -> Path:
        rid = report_id or f"rep_{uuid.uuid4().hex[:12]}"
        dest_path = self.sandbox.get_report_path(session_id, f"{rid}.html")
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Use relative link or data URL if evidence exists
        evidence_url = None
        if evidence_path and evidence_path.exists():
            evidence_url = f"/api/evidence/{evidence_path.name}?session_id={session_id}"

        html_content = self.template.render(
            report_id=rid,
            query=query,
            answer=answer,
            images=images,
            trace_view=trace_view,
            evidence_url=evidence_url,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        )

        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return dest_path
