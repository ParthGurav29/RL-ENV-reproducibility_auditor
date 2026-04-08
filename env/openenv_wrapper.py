"""
OpenEnv Spec-Compliant Wrapper
Wraps ReproducibilityAuditorEnv in the exact interface OpenEnv expects.
Uses Pydantic for schema validation on both observations and actions.

Multi-step support:
  Step 1 (triage) — TriageAction: identify suspicious files and categories
  Step 2 (audit)  — AuditAction:  full violation report with fixes
"""

from __future__ import annotations

import json
from typing import Any
from pydantic import BaseModel, Field
from env.base_env import ReproducibilityAuditorEnv, ALL_CATEGORIES


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class TriageAction(BaseModel):
    """Step 1 action — agent identifies suspicious files and violation categories."""

    suspicious_files: list[str] = Field(
        default_factory=list,
        description="File names the agent suspects contain reproducibility violations",
        min_length=0,
    )
    suspected_categories: list[str] = Field(
        default_factory=list,
        description=f"Violation categories the agent suspects are present. Valid categories: {', '.join(ALL_CATEGORIES)}",
        min_length=0,
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why these files/categories are suspicious",
    )


class ViolationObject(BaseModel):
    violation_type: str = Field(
        ...,
        description="The keyword or category of the violation (e.g. 'torch.manual_seed')",
    )
    file_name: str = Field(..., description="The specific file where it exists")
    line_number: int = Field(..., description="The specific line index")
    suggested_fix_code: str = Field(
        ..., description="The actual code snippet to be inserted or replaced"
    )


class AuditAction(BaseModel):
    """Step 2 action — the structured audit report an agent must submit."""

    violations: list[ViolationObject] = Field(
        default_factory=list,
        description="List of reproducibility violations found in the experiment",
        min_length=0,
    )
    reproducibility_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Agent's self-assessed reproducibility score (0=broken, 1=perfect)",
    )
    explanation: str = Field(
        default="",
        description="Human-readable summary of findings",
    )


class StepResult(BaseModel):
    """Returned by every call to step()."""

    observation: str
    reward: float = Field(gt=0.0, lt=1.0)
    terminated: bool
    truncated: bool
    done: bool = Field(
        default=False,
        description="True when episode is finished (terminated or truncated)",
    )
    info: dict[str, Any]


class ResetResult(BaseModel):
    """Returned by reset()."""

    observation: str
    info: dict[str, Any]


class EnvState(BaseModel):
    """Full serialisable state — required by OpenEnv spec.

    Extended with forensic metadata beyond basic step tracking:
      - episode_seed: the RNG seed used for presentation randomization
      - difficulty: the task's difficulty tier
      - num_files: how many files were loaded for this episode
      - focus_areas: what violation categories are relevant
      - is_episode_active: whether the episode is still in progress
      - current_step: which step of the episode we're on (0=reset, 1=triage done, 2=audit done)
    """

    task: str
    difficulty: str
    step_count: int
    max_steps: int
    current_step_type: str
    episode_seed: int | None
    num_files: int
    focus_areas: list[str]
    active_violations: list[str]
    is_episode_active: bool
    last_reward: float | None
    last_score_breakdown: dict[str, bool] | None
    triage_completed: bool


# ── Wrapper ───────────────────────────────────────────────────────────────────


class ReproducibilityEnvOpenEnv:
    """
    OpenEnv-compliant wrapper around ReproducibilityAuditorEnv.

    Key contract:
      reset()  → ResetResult
      step()   → StepResult         (action is TriageAction on step 1, AuditAction on step 2)
      state()  → EnvState
      spec()   → dict               (static metadata)
    """

    VERSION = "1.0.0"
    ENV_ID = "reproducibility-auditor-v1"

    def __init__(self, task: str = "easy"):
        self._inner = ReproducibilityAuditorEnv(task=task)
        self._task = task
        self._step_count = 0
        self._last_reward = None
        self._last_breakdown = None
        self._episode_seed = None
        self._active_violations: list[str] = []
        self._is_active = False
        self._triage_completed = False

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(self) -> ResetResult:
        obs, info = self._inner.reset()
        self._step_count = 0
        self._last_reward = None
        self._last_breakdown = None
        self._episode_seed = info.get("episode_seed")
        self._active_violations = info.get("active_violations", [])
        self._is_active = True
        self._triage_completed = False
        clean_info = {
            k: v
            for k, v in info.items()
            if k not in ("active_violations", "num_active_violations")
        }
        return ResetResult(observation=obs, info=clean_info)

    def step(self, action: str | dict | TriageAction | AuditAction) -> StepResult:
        import concurrent.futures

        # Auto-detect action type from dict keys (robust — no reliance on step count)
        is_triage = self._detect_triage(action)

        try:
            if is_triage:
                action_json = self._coerce_triage(action)
            else:
                action_json = self._coerce_action(action)
        except Exception as e:
            # Fallback: if action validation fails, use safe defaults
            if is_triage:
                action_json = TriageAction().model_dump_json()
            else:
                action_json = AuditAction().model_dump_json()

        # Execute step with 10 second timeout to prevent hanging
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._inner.step, action_json)
                obs, reward, terminated, truncated, info = future.result(timeout=10)
        except concurrent.futures.TimeoutError:
            # Step timed out - return safe failure result
            obs = "Step execution timed out after 10 seconds."
            reward = 0.01
            terminated = True
            truncated = True
            info = {"error": "Step timed out", "score_breakdown": {}}
        except Exception as e:
            # Any other error - return safe failure result
            obs = f"Step failed with error: {str(e)}"
            reward = 0.01
            terminated = True
            truncated = True
            info = {"error": str(e), "score_breakdown": {}}

        self._step_count += 1
        self._last_reward = reward

        if is_triage:
            self._triage_completed = True

        if info.get("score_breakdown"):
            self._last_breakdown = info.get("score_breakdown", {})

        if terminated:
            self._is_active = False

        return StepResult(
            observation=str(obs) if obs is not None else "",
            reward=round(max(0.01, min(0.99, reward)), 4),
            terminated=terminated,
            truncated=truncated,
            done=terminated or truncated,
            info=info,
        )

    def state(self) -> EnvState:
        meta = self._inner._current_metadata or {}
        step_type = "not_started"
        if self._is_active:
            step_type = "awaiting_triage" if self._step_count == 0 else "awaiting_audit"
        elif self._step_count >= 2:
            step_type = "completed"

        return EnvState(
            task=self._task,
            difficulty=meta.get("difficulty", self._task),
            step_count=self._step_count,
            max_steps=ReproducibilityAuditorEnv.MAX_STEPS,
            current_step_type=step_type,
            episode_seed=self._episode_seed,
            num_files=len(self._inner._current_files),
            focus_areas=meta.get("focus_areas", []),
            active_violations=self._active_violations,
            is_episode_active=self._is_active,
            last_reward=self._last_reward,
            last_score_breakdown=self._last_breakdown,
            triage_completed=self._triage_completed,
        )

    @staticmethod
    def spec() -> dict:
        return {
            "env_id": ReproducibilityEnvOpenEnv.ENV_ID,
            "version": ReproducibilityEnvOpenEnv.VERSION,
            "tasks": ["easy", "medium", "hard"],
            "observation_type": "text",
            "observation_schema": {
                "experiment_code": "Text — concatenated experiment source files with metadata header",
                "task_difficulty": "Text — easy | medium | hard",
                "num_files": "int — number of source files in the experiment",
                "focus_areas": "Text — comma-separated violation categories to check",
                "triage_feedback": "Text — (step 2 only) feedback from triage step confirming/rejecting agent's preliminary findings",
            },
            "action_type": "json",
            "action_schema": {
                "step_1_triage": TriageAction.model_json_schema(),
                "step_2_audit": AuditAction.model_json_schema(),
            },
            "reward_range": [0.01, 0.99],
            "episode_steps": 2,
            "step_descriptions": {
                "step_1": "Triage — identify suspicious files and violation categories. Receives feedback.",
                "step_2": "Audit — submit full violation report with fixes. Final grading.",
            },
            "randomization": "Violation-level: random subset of bugs injected per episode. Presentation-level: file order, decoy comments, whitespace.",
            "description": (
                "Agent audits a broken ML experiment and identifies reproducibility "
                "violations across seeds, dependencies, non-deterministic ops, and "
                "cross-file configuration gaps. Two-step episodes: (1) triage to "
                "identify suspicious files/categories with intermediate feedback, "
                "(2) full audit with detailed violation report. Each reset() produces "
                "a unique presentation of the same underlying bugs to prevent memorization."
            ),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_triage(action) -> bool:
        """Detect whether the action is a triage (step 1) or audit (step 2) based on content."""
        if isinstance(action, TriageAction):
            return True
        if isinstance(action, AuditAction):
            return False
        if isinstance(action, dict):
            # Triage actions have suspicious_files; audit actions have violations
            return "suspicious_files" in action or "suspected_categories" in action
        if isinstance(action, str):
            try:
                parsed = json.loads(action)
                return "suspicious_files" in parsed or "suspected_categories" in parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return False

    @staticmethod
    def _coerce_triage(action: str | dict | TriageAction | AuditAction) -> str:
        """Accept string, dict, or Pydantic model — return validated JSON string for triage."""
        if isinstance(action, TriageAction):
            return action.model_dump_json()
        if isinstance(action, AuditAction):
            # Convert AuditAction to TriageAction safely
            return TriageAction().model_dump_json()
        if isinstance(action, dict):
            try:
                validated = TriageAction(**action)
                return validated.model_dump_json()
            except:
                return TriageAction().model_dump_json()
        if isinstance(action, str):
            try:
                parsed = json.loads(action)
                validated = TriageAction(**parsed)
                return validated.model_dump_json()
            except:
                return TriageAction().model_dump_json()
        return TriageAction().model_dump_json()

    @staticmethod
    def _coerce_action(action: str | dict | TriageAction | AuditAction) -> str:
        """Accept string, dict, or Pydantic model — always return validated JSON string."""
        if isinstance(action, AuditAction):
            return action.model_dump_json()
        if isinstance(action, TriageAction):
            # Convert TriageAction to AuditAction safely
            return AuditAction().model_dump_json()
        if isinstance(action, dict):
            try:
                validated = AuditAction(**action)
                return validated.model_dump_json()
            except:
                return AuditAction().model_dump_json()
        if isinstance(action, str):
            try:
                parsed = json.loads(action)
                validated = AuditAction(**parsed)
                return validated.model_dump_json()
            except:
                return AuditAction().model_dump_json()
        return AuditAction().model_dump_json()
