"""
Easy Grader — checks for missing random seeds and unpinned dependencies.
Fully deterministic. Supports dynamic violation subsets with false-positive penalty.

Grading approach:
  - Keywords checked only in violation_type and suggested_fix_code fields
  - Bidirectional negation guard (25 chars before + after keyword)
  - file_name validated against expected file for each violation type
"""

import json

from env.graders.utils import is_valid_claim

SEED_VIOLATIONS = {
    "missing_random_seed":    "random.seed() not called",
    "missing_numpy_seed":     "np.random.seed() not called",
    "missing_torch_seed":     "torch.manual_seed() not called",
    "missing_cuda_seed":      "torch.cuda.manual_seed_all() not called",
}

UNPINNED_PACKAGES = [
    "torch",
    "numpy",
    "scikit-learn",
    "pandas",
]

EXTRA_CHECKS = [
    "missing_hashseed", "missing_cudnn_deterministic",
    "missing_cudnn_benchmark_off",
]

ALL_EASY_CHECKS = list(SEED_VIOLATIONS.keys()) + [f"unpinned_{p}" for p in UNPINNED_PACKAGES] + EXTRA_CHECKS

# Local file map for validation (avoids circular import with base_env)
_EASY_FILE_MAP = {
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
}


class EasyGrader:
    def score(self, action_json: str, active_violations: set[str] | None = None) -> tuple[float, dict]:
        if active_violations is None:
            active_violations = set(ALL_EASY_CHECKS)

        all_detections = {k: False for k in ALL_EASY_CHECKS}

        try:
            report = json.loads(action_json)
            violations_raw = report.get("violations", [])

            # Build per-claim entries with file_name for validation
            claim_entries = []  # list of (texts_for_keyword_check, file_name)
            if violations_raw and isinstance(violations_raw[0], dict):
                for v in violations_raw:
                    vtype = str(v.get("violation_type", "")).lower()
                    sfix = str(v.get("suggested_fix_code", "")).lower()
                    fname = str(v.get("file_name", "")).lower()
                    texts = [t for t in [vtype, sfix] if t]
                    if texts:
                        claim_entries.append((texts, fname))
            else:
                flat_texts = [str(v).lower() for v in violations_raw] + \
                             [str(f).lower() for f in report.get("fixes", [])]
                claim_entries = [(flat_texts, "")]
        except (json.JSONDecodeError, AttributeError):
            breakdown = {k: False for k in ALL_EASY_CHECKS if k in active_violations}
            return 0.0, breakdown

        # --- Seed checks ---
        self._check_detection(claim_entries, all_detections, "missing_random_seed", [
            "random.seed", "random seed", "random module",
            "python random", "import random", "stdlib random",
            "random.seed(", "missing random",
        ])
        self._check_detection(claim_entries, all_detections, "missing_numpy_seed", [
            "np.random.seed", "numpy seed", "numpy random",
            "numpy.random.seed", "numpy.random", "np random",
            "np.random", "missing numpy",
        ])
        self._check_detection(claim_entries, all_detections, "missing_torch_seed", [
            "torch.manual_seed", "torch seed", "manual_seed(",
            "missing torch seed", "torch manual seed",
        ])
        self._check_detection(claim_entries, all_detections, "missing_cuda_seed", [
            "cuda.manual_seed", "cuda seed", "manual_seed_all",
            "cuda manual seed", "gpu seed",
        ])

        # --- Dependency pinning checks ---
        for pkg in UNPINNED_PACKAGES:
            self._check_detection(claim_entries, all_detections, f"unpinned_{pkg}", [
                f"{pkg}==", f"pin {pkg}", f"{pkg} version", f"unpinned {pkg}"
            ])

        # --- Extra determinism checks ---
        self._check_detection(claim_entries, all_detections, "missing_hashseed", [
            "pythonhashseed", "hash seed", "hash randomization",
            "python hash seed", "hashseed",
        ])
        self._check_detection(claim_entries, all_detections, "missing_cudnn_deterministic", [
            "cudnn.deterministic", "cudnn deterministic", "cudnn_deterministic",
            "deterministic = true", "deterministic=true", "deterministic not set",
            "deterministic flag",
        ])
        self._check_detection(claim_entries, all_detections, "missing_cudnn_benchmark_off", [
            "cudnn.benchmark", "cudnn benchmark", "cudnn_benchmark",
            "benchmark = false", "benchmark=false", "benchmark not disabled",
            "benchmark not set", "benchmark flag", "benchmark enabled",
        ])

        # --- Scoring with false-positive penalty ---
        breakdown = {k: all_detections[k] for k in ALL_EASY_CHECKS if k in active_violations}
        hits = sum(1 for v in breakdown.values() if v)
        false_positives = sum(1 for k in ALL_EASY_CHECKS if k not in active_violations and all_detections[k])

        raw = hits - 1.0 * false_positives
        reward = round(max(0.0, min(1.0, raw / len(active_violations))), 4)
        return reward, breakdown

    def _check_detection(self, claim_entries, all_detections, violation_id, keywords):
        """Check if any claim matches keywords AND reports the correct file."""
        expected_file = _EASY_FILE_MAP.get(violation_id, "")
        for texts, fname in claim_entries:
            if is_valid_claim(texts, keywords):
                # File validation: if the claim specifies a file, it must match expected
                if fname and expected_file and fname != expected_file:
                    continue  # Wrong file — don't count this detection
                all_detections[violation_id] = True
                return