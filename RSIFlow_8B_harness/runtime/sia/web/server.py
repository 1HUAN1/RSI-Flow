"""FastAPI app + launchers for the runs visualizer.

``create_app(runs_dir)`` builds the app; ``serve(...)`` runs it in the
foreground (the ``sia web`` command); ``serve_in_background(...)`` starts it in a
daemon thread so the orchestrator can expose a live dashboard during ``sia run``.
"""

from __future__ import annotations

import threading
from pathlib import Path

from sia.logging_setup import get_logger
from sia.web import runs as runs_data

logger = get_logger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(runs_dir: str | Path):
    """Build the FastAPI application serving the runs under ``runs_dir``."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, PlainTextResponse
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The web visualizer needs FastAPI + uvicorn. Install with:\n    pip install fastapi uvicorn"
        ) from exc

    runs_root = Path(runs_dir).resolve()
    app = FastAPI(title="SIA Runs Visualizer", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.get("/api/runs")
    def api_runs():
        return runs_data.list_runs(runs_root)

    @app.get("/api/runs/{run_name}")
    def api_run(run_name: str):
        detail = runs_data.get_run(runs_root, run_name)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_name}")
        return detail

    @app.get("/api/runs/{run_name}/gens/{gen_name}/eval")
    def api_eval_details(run_name: str, gen_name: str):
        details = runs_data.get_eval_details(runs_root, run_name, gen_name)
        if details is None:
            raise HTTPException(status_code=404, detail="No evaluation details found")
        return details

    @app.get("/api/runs/{run_name}/gens/{gen_name}/artifact/{label}", response_class=PlainTextResponse)
    def api_artifact(run_name: str, gen_name: str, label: str):
        text = runs_data.get_artifact_text(runs_root, run_name, gen_name, label)
        if text is None:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {label}")
        return text

    @app.get("/api/runs/{run_name}/gens/{gen_name}/trajectory/{qid}")
    def api_trajectory(run_name: str, gen_name: str, qid: int):
        turns = runs_data.get_trajectory(runs_root, run_name, gen_name, qid)
        if turns is None:
            raise HTTPException(status_code=404, detail=f"Trajectory not found: q{qid}")
        return turns

    @app.get("/api/runs/{run_name}/gens/{gen_name}/openhands")
    def api_openhands_sessions(run_name: str, gen_name: str):
        sessions = runs_data.list_openhands_sessions(runs_root, run_name, gen_name)
        if sessions is None:
            raise HTTPException(status_code=404, detail="Generation not found")
        return sessions

    @app.get("/api/runs/{run_name}/gens/{gen_name}/openhands/{session}")
    def api_openhands_events(run_name: str, gen_name: str, session: str):
        events = runs_data.get_openhands_events(runs_root, run_name, gen_name, session)
        if events is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return events

    @app.get("/")
    def index():
        return FileResponse(_STATIC_DIR / "index.html")

    return app


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    runs_dir: str | Path = "./runs",
    open_browser: bool = True,
) -> None:
    """Run the server in the foreground (blocks). Used by ``sia web``."""
    import uvicorn

    app = create_app(runs_dir)
    url = f"http://{host}:{port}"
    logger.info(f"SIA visualizer serving {Path(runs_dir).resolve()} at {url}")
    if open_browser:
        _open_browser_later(url)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def serve_in_background(
    host: str = "127.0.0.1",
    port: int = 8000,
    runs_dir: str | Path = "./runs",
) -> threading.Thread | None:
    """Start the server in a daemon thread; never raises if deps are missing.

    Returns the thread (or ``None`` if the server could not start). Intended to
    give a live dashboard while ``sia run`` is executing.
    """
    try:
        import uvicorn

        app = create_app(runs_dir)
    except (RuntimeError, ModuleNotFoundError) as exc:
        logger.warning(f"Live dashboard unavailable: {exc}")
        return None

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    def _run() -> None:
        try:
            server.run()
        except Exception as exc:  # pragma: no cover - background best-effort
            logger.warning(f"Live dashboard stopped: {exc}")

    thread = threading.Thread(target=_run, name="sia-web", daemon=True)
    thread.start()
    logger.info(f"Live dashboard: http://{host}:{port} (serving {Path(runs_dir).resolve()})")
    return thread


def _open_browser_later(url: str, delay: float = 1.0) -> None:
    import webbrowser

    threading.Timer(delay, lambda: webbrowser.open(url)).start()
