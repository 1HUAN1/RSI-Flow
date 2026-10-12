"""Web visualizer for the RSIFlow ``runs/`` directory.

Public surface:
    create_app(runs_dir)          -> FastAPI app
    serve(...)                    -> run in foreground (``rsiflow web``)
    serve_in_background(...)      -> daemon thread (live dashboard during ``rsiflow run``)
"""

from rsiflow.web.server import create_app, serve, serve_in_background

__all__ = ["create_app", "serve", "serve_in_background"]
