"""app.py — Hugging Face Spaces + Docker entrypoint.

uvicorn is invoked as:  uvicorn app:app --host 0.0.0.0 --port 7860
so this module MUST export the FastAPI `app` object at module level.
"""
from server.app import app

__all__ = ["app"]
