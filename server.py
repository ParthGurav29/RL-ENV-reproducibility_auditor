"""
OpenEnv HTTP Server
Exposes the environment over HTTP so OpenEnv's eval harness can call it.
Endpoints follow the OpenEnv REST spec.

Extended with /leaderboard for persistent scoring history.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, field_validator

from env.openenv_wrapper import ReproducibilityEnvOpenEnv, AuditAction


# ── Global env instances (one per task) ──────────────────────────────────────

envs: dict[str, ReproducibilityEnvOpenEnv] = {}

# ── Leaderboard: persistent scoring history ──────────────────────────────────

leaderboard_entries: list[dict[str, Any]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    for task in ["easy", "medium", "hard"]:
        envs[task] = ReproducibilityEnvOpenEnv(task=task)
    yield


app = FastAPI(
    title="ML Reproducibility Auditor — OpenEnv",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",      # 👈 force enable
    redoc_url="/redoc"     # 👈 optional
)


# ── Request / response models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "easy"

    @field_validator("task")
    @classmethod
    def task_must_be_valid(cls, v: str) -> str:
        valid = {"easy", "medium", "hard"}
        if v not in valid:
            raise ValueError(f"task must be one of {sorted(valid)}, got {v!r}")
        return v


class StepRequest(BaseModel):
    task: str = "easy"
    action: dict[str, Any] = {}

    @field_validator("task")
    @classmethod
    def task_must_be_valid(cls, v: str) -> str:
        valid = {"easy", "medium", "hard"}
        if v not in valid:
            raise ValueError(f"task must be one of {sorted(valid)}, got {v!r}")
        return v


# ── Core Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Root ping endpoint — automated gate requires HTTP 200 at the Space URL."""
    return {
        "status": "ok",
        "env_id": "reproducibility-auditor-v1",
        "message": "OpenEnv Reproducibility Auditor is running. Use /reset to start.",
        "endpoints": ["/health", "/spec", "/reset", "/step", "/state", "/leaderboard"],
    }


@app.get("/spec")
def get_spec():
    """Return static environment metadata."""
    return ReproducibilityEnvOpenEnv.spec()


@app.get("/reset")
def reset_get(task: str = Query(default="easy", description="Task difficulty: easy | medium | hard")):
    """Reset via GET — validators may call GET /reset?task=easy."""
    if task not in envs:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task!r}. Must be one of: easy, medium, hard")
    result = envs[task].reset()
    return result.model_dump()


@app.post("/reset")
def reset_post(req: ResetRequest):
    """Reset via POST — inference.py and OpenEnv harness use this."""
    if req.task not in envs:
        raise HTTPException(status_code=400, detail=f"Unknown task: {req.task!r}. Must be one of: easy, medium, hard")
    result = envs[req.task].reset()
    return result.model_dump()


@app.post("/step")
async def step(req: StepRequest):
    """Submit an action and receive reward + info.

    Accepts TriageAction (step 1) or AuditAction (step 2) as the `action` dict.
    The result is also recorded to the leaderboard for scoring history.
    """
    if req.task not in envs:
        raise HTTPException(status_code=400, detail=f"Unknown task: {req.task!r}. Must be one of: easy, medium, hard")

    # Guard: if action is empty, return a 422 with a clear message
    if not req.action:
        raise HTTPException(
            status_code=422,
            detail="'action' field is required and must be a non-empty JSON object (TriageAction or AuditAction).",
        )

    try:
        result = envs[req.task].step(req.action)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Environment error: {e}")

    # Record to leaderboard
    breakdown = result.info.get("score_breakdown", {})
    checks_passed = sum(1 for v in breakdown.values() if v)
    checks_total = len(breakdown)

    leaderboard_entries.append({
        "task": req.task,
        "reward": result.reward,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return result.model_dump()


@app.get("/state")
def state(task: str = "easy"):
    """Return current environment state."""
    if task not in envs:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")
    return envs[task].state().model_dump()


# ── Leaderboard Endpoint ─────────────────────────────────────────────────────

@app.get("/leaderboard")
def leaderboard(task: str | None = Query(default=None, description="Filter by task (easy/medium/hard)")):
    """Return scoring history with per-task statistics.

    Optional query parameter `task` filters to a specific difficulty tier.
    Returns:
      - entries: list of all scored submissions (most recent first)
      - summary: per-task best score, average, and attempt count
    """
    entries = leaderboard_entries
    if task:
        entries = [e for e in entries if e["task"] == task]

    # Build per-task summary statistics
    summary: dict[str, dict[str, Any]] = {}
    for t in ["easy", "medium", "hard"]:
        task_entries = [e for e in leaderboard_entries if e["task"] == t]
        if task_entries:
            rewards = [e["reward"] for e in task_entries]
            summary[t] = {
                "attempts": len(task_entries),
                "best_score": round(max(rewards), 4),
                "avg_score": round(sum(rewards) / len(rewards), 4),
                "latest_score": round(rewards[-1], 4),
            }
        else:
            summary[t] = {
                "attempts": 0,
                "best_score": None,
                "avg_score": None,
                "latest_score": None,
            }

    # Overall average across best per-task scores
    best_scores = [s["best_score"] for s in summary.values() if s["best_score"] is not None]
    overall_best_avg = round(sum(best_scores) / len(best_scores), 4) if best_scores else None

    return {
        "total_submissions": len(leaderboard_entries),
        "overall_best_average": overall_best_avg,
        "per_task_summary": summary,
        "entries": list(reversed(entries)),  # Most recent first
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tasks_loaded": list(envs.keys()),
        "total_submissions": len(leaderboard_entries),
    }