from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import load_app_config, get_base_dir
from app.api.routes import router


def create_app() -> FastAPI:
    config = load_app_config()

    app = FastAPI(
        title="SatQuery AI API",
        description="Agentic Vision-Language Assistant for Multimodal Remote Sensing (SIH 2026 PS 26167 - ISRO)",
        version=config.system.version,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    app.include_router(router)

    # Mount frontend static files
    base_dir = get_base_dir()
    frontend_dir = base_dir / "frontend"
    frontend_dist = frontend_dir / "dist"

    if frontend_dist.exists() and (frontend_dist / "index.html").exists():
        assets_dir = frontend_dist / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
        app.mount("/static", StaticFiles(directory=str(frontend_dist)), name="static")

        @app.get("/", include_in_schema=False)
        def serve_index():
            return FileResponse(frontend_dist / "index.html")
    elif frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/", include_in_schema=False)
        def serve_index():
            index_file = frontend_dir / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
            return {"message": "SatQuery AI Backend Online"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host="127.0.0.1", port=8000, reload=True)
