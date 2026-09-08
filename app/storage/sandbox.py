import shutil
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from app.config import get_base_dir, load_app_config


class StorageSandbox:
    """Manages session isolation and path confinement for user sessions and artifacts."""

    def __init__(self, root_dir: Optional[Path] = None):
        if root_dir is None:
            config = load_app_config()
            self.root_dir = get_base_dir() / config.storage.root_dir
        else:
            self.root_dir = Path(root_dir)

        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        sdir = self._get_session_dir(sid)
        (sdir / "images").mkdir(parents=True, exist_ok=True)
        (sdir / "traces").mkdir(parents=True, exist_ok=True)
        (sdir / "evidence").mkdir(parents=True, exist_ok=True)
        (sdir / "reports").mkdir(parents=True, exist_ok=True)
        return sid

    def _get_session_dir(self, session_id: str) -> Path:
        # Prevent path traversal
        clean_sid = Path(session_id).name
        target = (self.root_dir / clean_sid).resolve()
        if not str(target).startswith(str(self.root_dir.resolve())):
            raise ValueError(f"Path traversal detected for session_id: {session_id}")
        return target

    def get_image_path(self, session_id: str, image_filename: str) -> Path:
        sdir = self._get_session_dir(session_id)
        clean_filename = Path(image_filename).name
        return sdir / "images" / clean_filename

    def get_trace_dir(self, session_id: str) -> Path:
        sdir = self._get_session_dir(session_id)
        return sdir / "traces"

    def get_evidence_path(self, session_id: str, evidence_filename: str) -> Path:
        sdir = self._get_session_dir(session_id)
        clean_name = Path(evidence_filename).name
        return sdir / "evidence" / clean_name

    def get_report_path(self, session_id: str, report_filename: str) -> Path:
        sdir = self._get_session_dir(session_id)
        clean_name = Path(report_filename).name
        return sdir / "reports" / clean_name

    def list_sessions(self) -> List[str]:
        if not self.root_dir.exists():
            return []
        return [d.name for d in self.root_dir.iterdir() if d.is_dir()]

    def session_exists(self, session_id: str) -> bool:
        sdir = self._get_session_dir(session_id)
        return sdir.exists() and sdir.is_dir()

    def delete_session(self, session_id: str) -> bool:
        sdir = self._get_session_dir(session_id)
        if sdir.exists():
            shutil.rmtree(sdir, ignore_errors=True)
            return True
        return False
