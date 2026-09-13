"""Security and robustness regression tests for the API, report, event, and evidence layers."""

import json
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

from app.api import routes
from app.data.ingestion import ImageMetadataEnvelope
from app.events.stream import EventStream, TraceView
from app.evidence.renderer import EvidenceRenderer
from app.reports.generator import ReportGenerator
from app.storage.sandbox import StorageSandbox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_envelope_dict(image_id, session_id="sess_other"):
    return {
        "image_id": image_id,
        "filename": f"{image_id}.png",
        "filepath": f"/tmp/{image_id}.png",
        "width": 16,
        "height": 16,
        "bands": 3,
        "dtype": "uint8",
        "file_size_bytes": 10,
        "sha256": "0" * 64,
        "modality": "optical",
    }


def _report_envelope():
    return ImageMetadataEnvelope(
        image_id="rep_img",
        filename='"><script>x</script>',
        filepath="/tmp/rep_img.png",
        width=16,
        height=16,
        bands=3,
        dtype="uint8",
        file_size_bytes=1,
        sha256="a" * 64,
        modality="optical",
    )


def _trace_view(session_id):
    return TraceView(
        trace_id="t1",
        session_id=session_id,
        query="q",
        created_at="2026-01-01T00:00:00Z",
        execution_trace=[],
    )


def _make_image(path):
    arr = np.zeros((24, 24, 3), dtype=np.uint8)
    arr[:, :, 1] = 160
    arr[0:6, :] = (230, 230, 230)
    Image.fromarray(arr).save(path)
    return path


def _render_envelope(tmp_path, image_id, modality="optical"):
    fp = Path(tmp_path) / f"{image_id}.png"
    _make_image(fp)
    return ImageMetadataEnvelope(
        image_id=image_id,
        filename=f"{image_id}.png",
        filepath=str(fp),
        width=24,
        height=24,
        bands=3,
        dtype="uint8",
        file_size_bytes=fp.stat().st_size,
        sha256="0" * 64,
        modality=modality,
    )


@pytest.fixture
def api_sandbox(tmp_path, monkeypatch):
    sb = StorageSandbox(root_dir=tmp_path / "sessions")
    monkeypatch.setattr(routes, "sandbox", sb)
    routes._ENVELOPE_CACHE.clear()
    sid = sb.create_session("sess_sec")
    other = sb.create_session("sess_other")
    return sb, sid, other


# ---------------------------------------------------------------------------
# 1. /api/trace traversal
# ---------------------------------------------------------------------------

TRAVERSAL_TRACE_IDS = [
    r"..\..\x",
    "..\\..\\x",
    "../x",
    "..",
    r"C:\Windows\win.ini",
    "C:/Windows/win.ini",
    r"\\server\share\evil",
    "/etc/passwd",
    "a:b",
]


@pytest.mark.parametrize("trace_id", TRAVERSAL_TRACE_IDS)
def test_trace_id_traversal_rejected(api_sandbox, trace_id):
    sb, sid, _ = api_sandbox
    with pytest.raises(HTTPException) as excinfo:
        routes.download_trace(trace_id, sid)
    assert excinfo.value.status_code in (400, 404)


def test_trace_id_cannot_escape_to_existing_file(api_sandbox):
    sb, sid, other = api_sandbox
    victim = sb.get_trace_dir(other) / "victim.jsonl"
    victim.write_text('{"secret": true}\n', encoding="utf-8")

    with pytest.raises(HTTPException) as excinfo:
        routes.download_trace(f"..\\..\\{other}\\traces\\victim", sid)
    assert excinfo.value.status_code in (400, 404)


def test_trace_valid_file_is_served(api_sandbox):
    sb, sid, _ = api_sandbox
    trace_path = sb.get_trace_dir(sid) / "tr_ok.jsonl"
    trace_path.write_text('{"ok": 1}\n', encoding="utf-8")

    response = routes.download_trace("tr_ok", sid)
    assert Path(response.path).resolve() == trace_path.resolve()


# ---------------------------------------------------------------------------
# 3. _get_envelope traversal / malformed JSON
# ---------------------------------------------------------------------------

def test_get_envelope_rejects_cross_session_traversal(api_sandbox):
    sb, sid, other = api_sandbox
    other_images = sb._get_session_dir(other) / "images"
    other_images.mkdir(parents=True, exist_ok=True)
    (other_images / "secret.json").write_text(
        json.dumps(_valid_envelope_dict("secret")), encoding="utf-8"
    )

    # Traversal attempts must not disclose the other session's envelope.
    assert routes._get_envelope(sid, f"..\\..\\{other}\\images\\secret") is None
    assert routes._get_envelope(sid, f"../{other}/images/secret") is None
    # The legitimate owner can still read it.
    assert routes._get_envelope(other, "secret") is not None


def test_describe_traversal_returns_404(api_sandbox):
    sb, sid, other = api_sandbox
    other_images = sb._get_session_dir(other) / "images"
    other_images.mkdir(parents=True, exist_ok=True)
    (other_images / "secret.json").write_text(
        json.dumps(_valid_envelope_dict("secret")), encoding="utf-8"
    )

    with pytest.raises(HTTPException) as excinfo:
        routes.describe_image(f"..\\..\\{other}\\images\\secret", sid)
    assert excinfo.value.status_code == 404


def test_get_envelope_non_envelope_json_returns_404(api_sandbox):
    sb, sid, _ = api_sandbox
    images = sb._get_session_dir(sid) / "images"
    images.mkdir(parents=True, exist_ok=True)
    (images / "notenv.json").write_text(json.dumps({"hello": "world"}), encoding="utf-8")

    assert routes._get_envelope(sid, "notenv") is None
    with pytest.raises(HTTPException) as excinfo:
        routes.describe_image("notenv", sid)
    assert excinfo.value.status_code == 404


def test_get_envelope_malformed_json_returns_404(api_sandbox):
    sb, sid, _ = api_sandbox
    images = sb._get_session_dir(sid) / "images"
    images.mkdir(parents=True, exist_ok=True)
    (images / "junk.json").write_text("{not valid json", encoding="utf-8")

    assert routes._get_envelope(sid, "junk") is None
    with pytest.raises(HTTPException) as excinfo:
        routes.describe_image("junk", sid)
    assert excinfo.value.status_code == 404


# ---------------------------------------------------------------------------
# 8. Directory cannot be served as an evidence file
# ---------------------------------------------------------------------------

def test_evidence_directory_returns_404(api_sandbox):
    sb, sid, _ = api_sandbox
    directory = sb.get_evidence_path(sid, "adir")
    directory.mkdir(parents=True, exist_ok=True)

    with pytest.raises(HTTPException) as excinfo:
        routes.download_evidence("adir", sid)
    assert excinfo.value.status_code == 404


# ---------------------------------------------------------------------------
# 2. Stored XSS in the HTML report
# ---------------------------------------------------------------------------

def test_report_escapes_script_and_quotes(tmp_path):
    sb = StorageSandbox(root_dir=tmp_path / "sessions")
    sid = sb.create_session("sess_report")
    generator = ReportGenerator(sb)

    script = "<script>alert(1)</script>"
    quoted = 'break"out'
    report_path = generator.generate_html_report(
        session_id=sid,
        query=script + quoted,
        answer=script,
        images=[_report_envelope()],
        trace_view=_trace_view(sid),
        report_id="rep_xss",
    )

    html = report_path.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&#34;" in html
    assert '"><script>' not in html


def test_report_url_encodes_session_id(tmp_path):
    sb = StorageSandbox(root_dir=tmp_path / "sessions")
    sid = sb.create_session("sess url")
    evidence = sb.get_evidence_path(sid, "e.png")
    evidence.write_bytes(b"\x89PNG\r\n")

    generator = ReportGenerator(sb)
    report_path = generator.generate_html_report(
        session_id=sid,
        query="q",
        answer="a",
        images=[_report_envelope()],
        trace_view=_trace_view(sid),
        evidence_path=evidence,
        report_id="rep_url",
    )

    html = report_path.read_text(encoding="utf-8")
    assert "session_id=sess%20url" in html


# ---------------------------------------------------------------------------
# 4. EventStream concurrency + robust serialization
# ---------------------------------------------------------------------------

def test_event_stream_concurrent_emits_are_not_corrupt(tmp_path):
    stream = EventStream(trace_id="t_conc", session_id="s_conc", log_dir=tmp_path)

    def worker():
        for i in range(100):
            stream.emit("TEST_EVENT", "ConcurrentStep", {"i": i})

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(stream.events) == 800

    lines = stream.log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 800

    event_ids = set()
    for line in lines:
        event = json.loads(line)  # raises on any truncated/corrupt line
        event_ids.add(event["event_id"])
    assert len(event_ids) == 800


def test_event_stream_serializes_numpy_payload(tmp_path):
    stream = EventStream(trace_id="t_np", session_id="s_np", log_dir=tmp_path)
    stream.emit(
        "NUMPY_EVENT",
        "Step",
        {"scalar": np.float32(1.5), "array": np.array([1, 2, 3]), "path": Path("x/y")},
    )

    line = stream.log_file.read_text(encoding="utf-8").strip()
    event = json.loads(line)
    assert event["payload"]["scalar"] == pytest.approx(1.5)
    assert event["payload"]["array"] == [1, 2, 3]
    assert event["payload"]["path"] == str(Path("x/y"))


# ---------------------------------------------------------------------------
# 5 & 7. Evidence renderer robustness
# ---------------------------------------------------------------------------

def test_filmstrip_malformed_boxes_do_not_raise(tmp_path):
    sb = StorageSandbox(root_dir=tmp_path / "sessions")
    sid = sb.create_session("sess_film")
    envelopes = [_render_envelope(tmp_path, f"e{i}") for i in range(3)]

    renderer = EvidenceRenderer(sb)
    out = renderer.render_temporal_sequence_filmstrip(
        sid, envelopes, boxes=[[5, 5, 15, 15], [1, 2, 3], [], [9, 9, 9, 9]]
    )
    assert out.is_file()


def test_filmstrip_fewer_boxes_than_transitions_does_not_raise(tmp_path):
    sb = StorageSandbox(root_dir=tmp_path / "sessions")
    sid = sb.create_session("sess_film2")
    envelopes = [_render_envelope(tmp_path, f"f{i}") for i in range(4)]

    renderer = EvidenceRenderer(sb)
    out = renderer.render_temporal_sequence_filmstrip(sid, envelopes, boxes=[[1, 1, 5, 5]])
    assert out.is_file()


def test_optical_sar_split_draws_boxes(tmp_path):
    sb = StorageSandbox(root_dir=tmp_path / "sessions")
    sid = sb.create_session("sess_fusion")
    opt = _render_envelope(tmp_path, "opt_1", modality="optical")
    sar = _render_envelope(tmp_path, "sar_1", modality="sar")

    renderer = EvidenceRenderer(sb)
    plain = renderer.render_optical_sar_split(sid, opt, sar, boxes=None).read_bytes()
    with_boxes = renderer.render_optical_sar_split(
        sid, opt, sar, boxes=[[4, 4, 18, 18]]
    ).read_bytes()
    assert with_boxes != plain

    # All-malformed boxes are skipped, so the output matches the plain render.
    malformed = renderer.render_optical_sar_split(
        sid, opt, sar, boxes=[[1, 2, 3]]
    ).read_bytes()
    assert malformed == plain