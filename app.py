"""app.py — Hugging Face Spaces + Docker entrypoint.

uvicorn is invoked as:  uvicorn app:app --host 0.0.0.0 --port 7860
so this module MUST export the FastAPI `app` object at module level.
"""
import uvicorn
from server import app  # re-export so 'uvicorn app:app' works

__all__ = ["app"]

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
