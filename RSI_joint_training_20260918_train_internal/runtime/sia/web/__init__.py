"""Web visualizer for the SIA ``runs/`` directory.

Public surface:
    create_app(runs_dir)          -> FastAPI app
    serve(...)                    -> run in foreground (``sia web``)
    serve_in_background(...)      -> daemon thread (live dashboard during ``sia run``)
"""

from sia.web.server import create_app, serve, serve_in_background

__all__ = ["create_app", "serve", "serve_in_background"]
