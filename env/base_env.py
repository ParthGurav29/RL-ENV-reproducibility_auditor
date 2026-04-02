"""
ML Experiment Reproducibility Auditor — RL Environment
Follows gymnasium API conventions for OpenEnv compatibility.

Multi-step episodes (2 steps):
  Step 1 — Triage: Agent identifies suspicious files and violation categories.
           Returns partial reward + actionable feedback for step 2.
  Step 2 — Audit:  Agent submits full violation report. Final grading.

Features:
  - Dynamic violation injection on every reset()
  - Presentation-level randomization (decoy comments, file order)
  - Multi-step trajectory with intermediate reward signal
  - Full render() support (observation state + grading results)
"""

import json
import random
import gymnasium as gym
from typing import Optional
from env.graders.easy_grader import EasyGrader
from env.graders.medium_grader import MediumGrader
from env.graders.hard_grader import HardGrader
from env.generators import select_violations, generate_files, ALL_VIOLATIONS_MAP

# ── Difficulty metadata ──────────────────────────────────────────────────────

TASK_METADATA = {
    "easy": {
        "difficulty": "easy",
        "focus_areas": ["random_seeds", "dependency_pinning"],
        "expected_violations": 11,
        "description": "Identify missing random seeds and unpinned package versions.",
    },
    "medium": {
        "difficulty": "medium",
        "focus_areas": ["pytorch_determinism", "dataloader_safety", "cudnn_flags"],
        "expected_violations": 8,
        "description": "Detect non-deterministic PyTorch operations and DataLoader issues.",
    },
    "hard": {
        "difficulty": "hard",
        "focus_areas": [
            "cross_file_seeds", "cublas_config", "version_conflicts",
            "weight_init", "config_override", "hash_randomization",
        ],
        "expected_violations": 7,
        "description": "Advanced cross-file audit: version conflicts, env vars, partial re-seeding.",
    },
}

# ── Violation → File mapping (which file each violation appears in) ──────────

VIOLATION_FILE_MAP = {
    # Easy
    "missing_random_seed": "train.py",
    "missing_numpy_seed": "train.py",
    "missing_torch_seed": "train.py",
    "missing_cuda_seed": "train.py",
    "unpinned_torch": "requirements.txt",
    "unpinned_numpy": "requirements.txt",
    "unpinned_scikit-learn": "requirements.txt",
    "unpinned_pandas": "requirements.txt",
    "missing_hashseed": "train.py",
    "missing_cudnn_deterministic": "train.py",
    "missing_cudnn_benchmark_off": "train.py",

    # Medium
    "dataloader_shuffle_no_seed": "train.py",
    "missing_deterministic_flag": "train.py",
    "missing_worker_seed": "train.py",
    "missing_generator_seed": "train.py",
    "missing_default_rng_seed": "train.py",
    "missing_dropout_seed": "train.py",
    # Hard
    "worker_seed_cross_file": "dataset.py",
    "cublas_workspace_config": "train.py",
    "package_version_conflict": "requirements.txt",
    "model_weight_init_seed": "model.py",
    "config_yaml_override": "train.py",
    "hash_randomization": "train.py",
    "multiprocessing_no_seed": "train.py",
}

# ── Violation → Category mapping ─────────────────────────────────────────────

VIOLATION_CATEGORY_MAP = {
    # Seeds
    "missing_random_seed": "random_seeds",
    "missing_numpy_seed": "random_seeds",
    "missing_torch_seed": "random_seeds",
    "missing_cuda_seed": "random_seeds",
    # Dependencies
    "unpinned_torch": "dependency_pinning",
    "unpinned_numpy": "dependency_pinning",
    "unpinned_scikit-learn": "dependency_pinning",
    "unpinned_pandas": "dependency_pinning",
    "package_version_conflict": "dependency_pinning",
    # Determinism flags
    "missing_cudnn_deterministic": "determinism_flags",
    "missing_cudnn_benchmark_off": "determinism_flags",
    "missing_deterministic_flag": "determinism_flags",
    # Environment variables
    "missing_hashseed": "environment_config",
    "hash_randomization": "environment_config",

    "cublas_workspace_config": "environment_config",

    # DataLoader
    "dataloader_shuffle_no_seed": "dataloader_reproducibility",
    "missing_worker_seed": "dataloader_reproducibility",
    "missing_generator_seed": "dataloader_reproducibility",
    "worker_seed_cross_file": "dataloader_reproducibility",
    # Model initialization
    "model_weight_init_seed": "model_initialization",
    "missing_dropout_seed": "model_initialization",
    # Advanced
    "config_yaml_override": "configuration_management",
    "multiprocessing_no_seed": "multiprocessing",
    "missing_default_rng_seed": "rng_initialization",
}

# All valid categories an agent can claim
ALL_CATEGORIES = sorted(set(VIOLATION_CATEGORY_MAP.values()))

# ── Decoy comments injected randomly ────────────────────────────────────────

_DECOY_COMMENTS = [
    "# Performance note: batch size tuned for V100 GPU",
    "# Ref: https://arxiv.org/abs/2002.05709",
    "# TODO: add learning rate warmup",
    "# This follows the setup from the original paper",
    "# Alternative: use AdamW with weight decay 0.01",
    "# Data loading pipeline benchmarked at 3.2k samples/s",
    "# Checkpoint saved every 5 epochs",
    "# Gradient accumulation disabled for simplicity",
    "# Mixed precision disabled — see known issues",
]


class ReproducibilityAuditorEnv(gym.Env):
    """
    Multi-step RL environment for ML reproducibility auditing.

    Observation: Structured text containing experiment source files.
    Action:      Step 1 — TriageAction JSON.  Step 2 — AuditAction JSON.
    Reward:      Step 1 — triage F1 score.    Step 2 — violation detection score.
    Episodes:    Two-step — triage → feedback → audit → done.
    """

    metadata = {"render_modes": ["human"]}
    MAX_STEPS = 2

    GRADER_MAP = {
        "easy":   EasyGrader,
        "medium": MediumGrader,
        "hard":   HardGrader,
    }

    def __init__(self, task: str = "easy", render_mode: Optional[str] = None):
        super().__init__()
        assert task in self.GRADER_MAP, f"task must be one of {list(self.GRADER_MAP.keys())}"
        self.task = task
        self.render_mode = render_mode

        self.grader = self.GRADER_MAP[task]()

        # Observation space — plain text (concatenated experiment source files)
        self.observation_space = gym.spaces.Text(min_length=0, max_length=100_000)

        # Action: text (JSON-serialised TriageAction or AuditAction)
        self.action_space = gym.spaces.Text(min_length=0, max_length=10_000)

        self._current_obs: Optional[str] = None
        self._current_files: dict[str, str] = {}
        self._current_metadata: dict = {}
        self._active_violations: set[str] = set()
        self._episode_seed: Optional[int] = None
        self._step_count: int = 0
        self._last_reward: Optional[float] = None
        self._last_breakdown: Optional[dict] = None
        self._triage_feedback: Optional[dict] = None

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Derive a per-episode seed for synthetic randomization
        if seed is not None:
            self._episode_seed = seed
        else:
            self._episode_seed = random.randint(0, 2**31)

        rng = random.Random(self._episode_seed)

        # Randomly select which violations are active this episode
        self._active_violations = select_violations(self.task, rng)

        # Generate task files with only the active violations present
        raw_files = generate_files(self.task, self._active_violations, rng)

        # Apply presentation-level randomization (decoy comments, file order)
        randomized_files = self._randomize_presentation(raw_files, rng)

        self._current_files = randomized_files
        self._current_metadata = TASK_METADATA[self.task]
        self._current_obs = self._format_observation(randomized_files, self.task)
        self._step_count = 0
        self._last_reward = None
        self._last_breakdown = None
        self._triage_feedback = None

        info = {
            "task": self.task,
            "num_files": len(randomized_files),
            "difficulty": self._current_metadata["difficulty"],
            "focus_areas": self._current_metadata["focus_areas"],
            "episode_seed": self._episode_seed,
            "active_violations": sorted(self._active_violations),
            "num_active_violations": len(self._active_violations),
            "max_steps": self.MAX_STEPS,
        }

        if self.render_mode == "human":
            self._render_observation()

        return self._current_obs, info

    # ------------------------------------------------------------------
    def step(self, action: str):
        """
        Multi-step dispatch — detects action type from JSON keys:
          TriageAction (has 'suspicious_files') → triage step → feedback + partial reward
          AuditAction  (has 'violations')       → audit step  → final score
        """
        self._step_count += 1

        # Detect action type from content (robust against state desync)
        try:
            parsed = json.loads(action)
            is_triage = "suspicious_files" in parsed or "suspected_categories" in parsed
        except (json.JSONDecodeError, TypeError):
            is_triage = False

        if is_triage:
            return self._triage_step(action)
        else:
            return self._audit_step(action)

    # ------------------------------------------------------------------
    # Step 1: Triage
    # ------------------------------------------------------------------

    def _triage_step(self, action: str):
        """Agent identifies suspicious files and violation categories. Gets feedback."""
        try:
            triage = json.loads(action)
        except (json.JSONDecodeError, TypeError):
            triage = {"suspicious_files": [], "suspected_categories": [], "reasoning": ""}

        triage_reward, feedback = self._score_triage(triage)

        # Build enhanced observation with triage feedback for step 2
        feedback_text = self._format_triage_feedback(feedback)
        enhanced_obs = self._current_obs + "\n\n" + feedback_text
        self._current_obs = enhanced_obs  # Agent sees this in step 2

        self._last_reward = triage_reward
        self._triage_feedback = feedback

        info = {
            "task": self.task,
            "step": 1,
            "step_type": "triage",
            "triage_reward": triage_reward,
            "triage_feedback": feedback,
            "max_steps": self.MAX_STEPS,
        }

        if self.render_mode == "human":
            self._render_triage(triage_reward, feedback)

        return enhanced_obs, triage_reward, False, False, info

    def _score_triage(self, triage: dict) -> tuple[float, dict]:
        """Score the triage action on file and category identification."""
        # Ground truth: which files actually contain violations
        true_files = set()
        for v in self._active_violations:
            if v in VIOLATION_FILE_MAP:
                true_files.add(VIOLATION_FILE_MAP[v])

        # Ground truth: which categories are active
        true_categories = set()
        for v in self._active_violations:
            if v in VIOLATION_CATEGORY_MAP:
                true_categories.add(VIOLATION_CATEGORY_MAP[v])

        # Agent's claims
        claimed_files = set(str(f).strip() for f in triage.get("suspicious_files", []))
        claimed_cats = set(str(c).strip().lower() for c in triage.get("suspected_categories", []))
        # Normalize category names (allow underscores or spaces)
        true_cats_normalized = {c.lower() for c in true_categories}

        # --- File scoring (F1) ---
        file_hits = len(claimed_files & true_files)
        file_fps = len(claimed_files - true_files)
        file_recall = file_hits / len(true_files) if true_files else 0.0
        file_precision = file_hits / len(claimed_files) if claimed_files else 0.0
        file_f1 = (2 * file_precision * file_recall / (file_precision + file_recall)
                   if (file_precision + file_recall) > 0 else 0.0)

        # --- Category scoring (F1) ---
        cat_hits = len(claimed_cats & true_cats_normalized)
        cat_fps = len(claimed_cats - true_cats_normalized)
        cat_recall = cat_hits / len(true_cats_normalized) if true_cats_normalized else 0.0
        cat_precision = cat_hits / len(claimed_cats) if claimed_cats else 0.0
        cat_f1 = (2 * cat_precision * cat_recall / (cat_precision + cat_recall)
                  if (cat_precision + cat_recall) > 0 else 0.0)

        # Combined triage reward (file identification weighted higher — it's harder)
        triage_reward = round(0.6 * file_f1 + 0.4 * cat_f1, 4)

        # Build feedback dict — tells agent which of its claims were correct
        feedback = {
            "files_confirmed": sorted(claimed_files & true_files),
            "files_rejected": sorted(claimed_files - true_files),
            "num_violation_files": len(true_files),
            "categories_confirmed": sorted(claimed_cats & true_cats_normalized),
            "categories_rejected": sorted(claimed_cats - true_cats_normalized),
            "num_violation_categories": len(true_cats_normalized),
            "triage_score": triage_reward,
            "hint": f"Violations appear in {len(true_files)} file(s) across {len(true_cats_normalized)} categories. Focus your detailed audit on confirmed files.",
        }

        return triage_reward, feedback

    @staticmethod
    def _format_triage_feedback(feedback: dict) -> str:
        """Format triage feedback as text to append to the observation."""
        lines = [
            "=== TRIAGE FEEDBACK (from Step 1) ===",
            f"Triage score: {feedback['triage_score']:.4f}",
            "",
        ]
        if feedback["files_confirmed"]:
            lines.append(f"✅ Confirmed suspicious files: {', '.join(feedback['files_confirmed'])}")
        if feedback["files_rejected"]:
            lines.append(f"❌ Rejected files (no violations found): {', '.join(feedback['files_rejected'])}")
        lines.append(f"ℹ️  Total files with violations: {feedback['num_violation_files']}")
        lines.append("")
        if feedback["categories_confirmed"]:
            lines.append(f"✅ Confirmed categories: {', '.join(feedback['categories_confirmed'])}")
        if feedback["categories_rejected"]:
            lines.append(f"❌ Rejected categories: {', '.join(feedback['categories_rejected'])}")
        lines.append(f"ℹ️  Total active categories: {feedback['num_violation_categories']}")
        lines.append("")
        lines.append(f"💡 {feedback['hint']}")
        lines.append("=" * 40)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Step 2: Full Audit (existing logic)
    # ------------------------------------------------------------------

    def _audit_step(self, action: str):
        """Standard grading step — score the full violation audit report."""
        reward, breakdown = self.grader.score(action, self._active_violations)
        self._last_reward = reward
        self._last_breakdown = breakdown

        terminated = True   # Episode done after step 2
        truncated = False
        info = {
            "task": self.task,
            "step": 2,
            "step_type": "audit",
            "score_breakdown": breakdown,
            "final_reward": reward,
            "triage_feedback": self._triage_feedback,
        }

        if self.render_mode == "human":
            self._render_grading(reward, breakdown)

        return self._current_obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        """Standard gymnasium render() — shows current environment state."""
        if self.render_mode != "human":
            return

        if self._last_reward is not None:
            self._render_grading(self._last_reward, self._last_breakdown or {})
        else:
            self._render_observation()

    def _render_observation(self):
        """Print a summary of the current observation (experiment under audit)."""
        meta = self._current_metadata
        print(f"\n{'━' * 60}")
        print(f"  📂 TASK: {self.task.upper()}  |  Difficulty: {meta['difficulty']}")
        print(f"  📄 Files loaded: {len(self._current_files)}")
        print(f"  🎯 Focus areas: {', '.join(meta['focus_areas'])}")
        print(f"  🔢 Active violations: {len(self._active_violations)} / {meta['expected_violations']}")
        print(f"  🌱 Episode seed: {self._episode_seed}")
        print(f"  📊 Episode steps: {self.MAX_STEPS} (triage → audit)")
        print(f"{'━' * 60}")
        for name in self._current_files:
            lines = self._current_files[name].count('\n') + 1
            print(f"    📄 {name} ({lines} lines)")
        print(f"{'━' * 60}\n")

    @staticmethod
    def _render_triage(reward: float, feedback: dict):
        """Print triage results after step 1."""
        print(f"\n{'─' * 60}")
        print(f"  🔍 TRIAGE REWARD: {reward:.4f}")
        if feedback.get("files_confirmed"):
            print(f"  ✅ Confirmed files: {', '.join(feedback['files_confirmed'])}")
        if feedback.get("files_rejected"):
            print(f"  ❌ Rejected files: {', '.join(feedback['files_rejected'])}")
        if feedback.get("categories_confirmed"):
            print(f"  ✅ Confirmed categories: {', '.join(feedback['categories_confirmed'])}")
        if feedback.get("categories_rejected"):
            print(f"  ❌ Rejected categories: {', '.join(feedback['categories_rejected'])}")
        print(f"  💡 {feedback.get('hint', '')}")
        print(f"{'─' * 60}\n")

    @staticmethod
    def _render_grading(reward: float, breakdown: dict):
        """Print grading results after step 2."""
        print(f"\n{'═' * 60}")
        print(f"  🏆 REWARD: {reward:.4f}")
        print(f"  📊 BREAKDOWN:")
        for check, passed in breakdown.items():
            icon = "✅" if passed else "❌"
            print(f"    {icon}  {check}")
        print(f"{'═' * 60}\n")

    # ------------------------------------------------------------------
    # Presentation randomization
    # ------------------------------------------------------------------

    @staticmethod
    def _randomize_presentation(files: dict[str, str], rng: random.Random) -> dict[str, str]:
        """
        Apply presentation-level randomization to make every episode unique
        while preserving all reproducibility bugs for the grader.

        Techniques:
          1. Shuffle the file presentation order
          2. Inject random decoy comments into source files
          3. Randomly swap harmless whitespace patterns
        """
        # 1. Shuffle file order
        file_items = list(files.items())
        rng.shuffle(file_items)

        randomized = {}
        for name, content in file_items:
            lines = content.split('\n')

            # 2. Inject 1-3 decoy comments at random positions (only in .py files)
            if name.endswith('.py'):
                num_decoys = rng.randint(1, 3)
                decoys = rng.sample(_DECOY_COMMENTS, min(num_decoys, len(_DECOY_COMMENTS)))
                for decoy in decoys:
                    # Insert after imports or at random safe positions
                    # Find a blank-line position to inject (avoid breaking syntax)
                    safe_positions = [
                        i for i, line in enumerate(lines)
                        if line.strip() == '' and i > 0
                    ]
                    if safe_positions:
                        pos = rng.choice(safe_positions)
                        lines.insert(pos, decoy)

            # 3. Random trailing whitespace variation
            if rng.random() < 0.5:
                lines = [line.rstrip() for line in lines]

            randomized[name] = '\n'.join(lines)

        return randomized

    @staticmethod
    def _format_observation(files: dict[str, str], task: str) -> str:
        """Build the text observation with structured metadata header."""
        meta = TASK_METADATA[task]
        parts = [
            "=== ML EXPERIMENT FILES ===",
            f"Task: {task} | Difficulty: {meta['difficulty']}",
            f"Focus areas: {', '.join(meta['focus_areas'])}",
            f"Number of files: {len(files)}",
            f"Episode type: 2-step (triage → audit)",
            f"Available categories: {', '.join(ALL_CATEGORIES)}",
            "",
        ]
        for name, content in files.items():
            parts.append(f"\n--- FILE: {name} ---\n{content}")
        return "\n".join(parts)