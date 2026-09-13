import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from app.config import load_app_config
from app.storage.sandbox import StorageSandbox
from app.data.ingestion import ImageIngestionService, ImageMetadataEnvelope
from app.data.pair_validator import PairValidator, ValidationError
from app.registry.store import ToolRegistryStore
from app.runtime.manager import ModelRuntimeManager
from app.agent.controller import AgentController, QueryExecutionResult
from app.agent.classifier import TaskClassifier


from app.data.preprocessing import PreprocessingService

router = APIRouter(prefix="/api")

sandbox = StorageSandbox()
ingestion_svc = ImageIngestionService()
prep_svc = PreprocessingService()
runtime_mgr = ModelRuntimeManager()
registry_store = ToolRegistryStore(runtime_mgr=runtime_mgr)
agent_controller = AgentController(registry=registry_store, sandbox=sandbox)

# In-memory session envelope cache (also persisted to disk in session)
_ENVELOPE_CACHE: Dict[str, ImageMetadataEnvelope] = {}


def _save_envelope(session_id: str, envelope: ImageMetadataEnvelope):
    _ENVELOPE_CACHE[f"{session_id}:{envelope.image_id}"] = envelope
    sdir = sandbox._get_session_dir(session_id)
    meta_path = sdir / "images" / f"{envelope.image_id}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(envelope.model_dump_json(indent=2))


def _get_envelope(session_id: str, image_id: str) -> Optional[ImageMetadataEnvelope]:
    # Reject traversal outright (e.g. "..\\..\\other_session\\images\\id") rather
    # than letting Path(...).name silently strip the directory components.
    try:
        clean_id = sandbox.sanitize_component(image_id, "image_id")
    except ValueError:
        return None

    key = f"{session_id}:{clean_id}"
    if key in _ENVELOPE_CACHE:
        return _ENVELOPE_CACHE[key]

    try:
        sdir = sandbox._get_session_dir(session_id)
        images_dir = (sdir / "images").resolve()
        meta_path = (images_dir / f"{clean_id}.json").resolve()
        if not meta_path.is_relative_to(images_dir):
            return None
        if not meta_path.is_file():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        env = ImageMetadataEnvelope(**data)
    except (ValueError, PydanticValidationError, OSError, json.JSONDecodeError):
        # Missing file, malformed JSON, or valid JSON that is not an envelope.
        return None

    _ENVELOPE_CACHE[key] = env
    return env


# --- Request / Response Models ---
class QueryRequest(BaseModel):
    session_id: str
    query: str
    image_ids: List[str]
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ValidateRequest(BaseModel):
    session_id: str
    image_ids: List[str]
    task: Optional[str] = None
    query: Optional[str] = None


class SpectralIndexRequest(BaseModel):
    session_id: str
    image_id: str


# --- Endpoints ---

@router.get("/health")
def get_health():
    config = load_app_config()
    return {
        "status": "HEALTHY",
        "service": config.system.name,
        "version": config.system.version,
        "runtime": runtime_mgr.get_runtime_status(),
        "checkpoint_artifacts": runtime_mgr.get_artifact_status(),
        "tool_registry_version": registry_store.version,
        "total_tools": len(registry_store.list_descriptors())
    }


@router.get("/registry")
def get_registry():
    return {
        "registry_version": registry_store.version,
        "tools": registry_store.list_descriptors()
    }


@router.get("/benchmarks")
def get_benchmarks():
    bench_file = Path("eval/benchmark_results.json")
    if bench_file.exists():
        try:
            with open(bench_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    from eval.benchmark_evaluator import BenchmarkEvaluator
    evaluator = BenchmarkEvaluator()
    return evaluator.run_all()


@router.post("/sessions")
def create_session():
    sid = sandbox.create_session()
    return {"session_id": sid, "status": "CREATED"}


@router.get("/sessions")
def list_sessions():
    return {"sessions": sandbox.list_sessions()}


@router.post("/ingest")
async def ingest_images(
    session_id: Optional[str] = Form(None),
    modality: Optional[str] = Form(None),
    files: List[UploadFile] = File(...)
):
    if not session_id or not sandbox.session_exists(session_id):
        session_id = sandbox.create_session(session_id)

    envelopes: List[ImageMetadataEnvelope] = []

    for file in files:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail=f"Uploaded file '{file.filename}' is empty.")

        dest_path = sandbox.get_image_path(session_id, file.filename)
        envelope = ingestion_svc.ingest_image(
            file_bytes=file_bytes,
            filename=file.filename,
            dest_path=dest_path,
            forced_modality=modality
        )
        _save_envelope(session_id, envelope)
        envelopes.append(envelope)

    return {
        "session_id": session_id,
        "count": len(envelopes),
        "images": envelopes
    }


@router.post("/validate")
def validate_inputs(req: ValidateRequest):
    envelopes = []
    for iid in req.image_ids:
        env = _get_envelope(req.session_id, iid)
        if not env:
            raise HTTPException(status_code=404, detail=f"Image ID '{iid}' not found in session '{req.session_id}'.")
        envelopes.append(env)

    if not envelopes:
        return {"valid": False, "error_code": "VAL_NO_INPUT_IMAGES", "message": "No images provided for validation."}

    # Count-based modality pairing sanity check (unchanged behavior).
    def _check_pairing():
        if len(envelopes) == 1:
            PairValidator.validate_single(envelopes[0])
            return "single"
        elif len(envelopes) == 2:
            # Test cross-modal or bi-temporal
            try:
                PairValidator.validate_cross_modal_pair(envelopes[0], envelopes[1])
                return "cross_modal_opt_sar"
            except ValidationError:
                PairValidator.validate_bi_temporal_pair(envelopes[0], envelopes[1])
                return "bi_temporal"
        return None

    try:
        if req.query and req.query.strip():
            # Query-aware validation: classify intent independent of image count and
            # enforce the task's observation requirements in addition to pairing.
            from app.agent.validator import AgentInputValidator
            task, _ = TaskClassifier.classify(req.query, envelopes)
            AgentInputValidator.validate(task, envelopes)
            pair_type = _check_pairing()
            result = {"valid": True, "task": task, "images_checked": len(envelopes)}
            if pair_type:
                result["pair_type"] = pair_type
            return result
        if req.task:
            from app.agent.validator import AgentInputValidator
            AgentInputValidator.validate(req.task, envelopes)
            return {"valid": True, "task": req.task, "images_checked": len(envelopes)}
        else:
            pair_type = _check_pairing()
            if pair_type:
                return {"valid": True, "pair_type": pair_type}
    except ValidationError as ve:
        return {"valid": False, "error_code": ve.code, "message": ve.message}

    return {"valid": True}


@router.get("/describe/{image_id}")
def describe_image(image_id: str, session_id: str = Query(...)):
    env = _get_envelope(session_id, image_id)
    if not env:
        raise HTTPException(status_code=404, detail=f"Image ID '{image_id}' not found in session '{session_id}'.")
    return env


@router.post("/query", response_model=QueryExecutionResult)
def execute_query(req: QueryRequest):
    if not sandbox.session_exists(req.session_id):
        raise HTTPException(status_code=404, detail=f"Session ID '{req.session_id}' not found.")

    envelopes = []
    for iid in req.image_ids:
        env = _get_envelope(req.session_id, iid)
        if not env:
            raise HTTPException(status_code=404, detail=f"Image ID '{iid}' not found in session '{req.session_id}'.")
        envelopes.append(env)

    if not envelopes:
        raise HTTPException(status_code=400, detail="At least one input image is required to execute a query.")

    try:
        result = agent_controller.execute_query(
            session_id=req.session_id,
            query=req.query,
            images=envelopes,
            parameters=req.parameters
        )
        return result
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail={"code": ve.code, "message": ve.message})
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error while executing query.")


@router.get("/evidence/{filename}")
def download_evidence(filename: str, session_id: str = Query(...)):
    path = sandbox.get_evidence_path(session_id, filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Evidence file '{filename}' not found.")
    media_type = "application/geo+json" if path.suffix == ".geojson" else "image/png"
    return FileResponse(path, media_type=media_type)



@router.get("/report/{filename}")
def download_report(filename: str, session_id: str = Query(...)):
    path = sandbox.get_report_path(session_id, filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Report file '{filename}' not found.")
    return FileResponse(path, media_type="text/html")


@router.get("/trace/{trace_id}")
def download_trace(trace_id: str, session_id: str = Query(...)):
    try:
        path = sandbox.get_trace_path(session_id, trace_id)
    except (ValueError, OSError):
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found.")
    return FileResponse(path, media_type="application/x-jsonlines")


@router.post("/spectral-indices")
def get_spectral_indices(req: SpectralIndexRequest):
    env = _get_envelope(req.session_id, req.image_id)
    if not env:
        raise HTTPException(status_code=404, detail=f"Image ID '{req.image_id}' not found in session '{req.session_id}'.")

    img_path = Path(env.filepath)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Underlying image file not found.")

    data, tags = ingestion_svc.read_image_data(img_path)
    indices = prep_svc.compute_spectral_indices(data, modality=env.modality)
    return {
        "image_id": req.image_id,
        "modality": env.modality,
        "indices": indices
    }
