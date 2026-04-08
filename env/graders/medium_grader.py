"""
Medium Grader — detects non-deterministic operations in the training loop.
Keywords checked only in violation_type and suggested_fix_code fields.
Bidirectional negation guard (25 chars before + after keyword).
file_name validated against expected file for each violation type.
"""

import json

from env.graders.utils import is_valid_claim

NONDETERMINISM_CHECKS = {
    "dataloader_shuffle_no_seed": "DataLoader uses shuffle=True but no generator seed is set",
    "missing_deterministic_flag": "torch.use_deterministic_algorithms(True) not set",
    "missing_cudnn_deterministic": "torch.backends.cudnn.deterministic not set to True",
    "missing_cudnn_benchmark_off": "torch.backends.cudnn.benchmark not disabled",
    "missing_worker_seed": "DataLoader worker_init_fn not set",
    "missing_generator_seed": "torch.Generator() created without a seed",
    "missing_default_rng_seed": "np.random.default_rng() used without a seed argument",
    "missing_dropout_seed": "Dropout layer used without seed guard for determinism",
}

ALL_MEDIUM_CHECKS = list(NONDETERMINISM_CHECKS.keys())

# Local file map for validation (avoids circular import with base_env)
_MEDIUM_FILE_MAP = {
    "dataloader_shuffle_no_seed": "train.py",
    "missing_deterministic_flag": "train.py",
    "missing_cudnn_deterministic": "train.py",
    "missing_cudnn_benchmark_off": "train.py",
    "missing_worker_seed": "train.py",
    "missing_generator_seed": "train.py",
    "missing_default_rng_seed": "train.py",
    "missing_dropout_seed": "train.py",
}


class MediumGrader:
    def score(self, action_json: str, active_violations: set[str] | None = None) -> tuple[float, dict]:
        if active_violations is None:
            active_violations = set(ALL_MEDIUM_CHECKS)

        all_detections = {k: False for k in ALL_MEDIUM_CHECKS}

        try:
            report = json.loads(action_json)
            violations_raw = report.get("violations", [])
            claim_entries = _extract_claim_entries(violations_raw, report)
        except (json.JSONDecodeError, AttributeError):
            breakdown = {k: False for k in ALL_MEDIUM_CHECKS if k in active_violations}
            return 0.01, breakdown

        self._check_detection(claim_entries, all_detections, "dataloader_shuffle_no_seed", [
            "shuffle", "dataloader", "shuffle=true",
            "shuffle = true", "shuffle=True",
        ])
        self._check_detection(claim_entries, all_detections, "missing_deterministic_flag", [
            "use_deterministic_algorithms", "deterministic algorithms",
            "deterministic_algorithms", "torch.use_deterministic",
            "deterministic_algorithms missing",
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
        self._check_detection(claim_entries, all_detections, "missing_worker_seed", [
            "worker_init_fn", "worker seed", "dataloader worker",
            "worker init", "worker_init",
        ])
        self._check_detection(claim_entries, all_detections, "missing_generator_seed", [
            "torch.generator", "generator seed", "generator().manual_seed",
            "generator()", "generator manual_seed", "torch.generator()",
        ])
        self._check_detection(claim_entries, all_detections, "missing_default_rng_seed", [
            "default_rng", "np.random.default_rng",
            "numpy.random.default_rng", "default rng",
        ])
        self._check_detection(claim_entries, all_detections, "missing_dropout_seed", [
            "dropout without seed", "nn.dropout without seed",
            "dropout seed guard", "dropout determinism",
            "dropout not seeded", "dropout randomness",
            "dropout layer not seeded", "nn.dropout missing determinism",
            "dropout without determinism", "dropout non-deterministic",
            "dropout not deterministic",
        ])

        breakdown = {k: all_detections[k] for k in ALL_MEDIUM_CHECKS if k in active_violations}
        hits = sum(1 for v in breakdown.values() if v)
        false_positives = sum(1 for k in ALL_MEDIUM_CHECKS if k not in active_violations and all_detections[k])
        raw = hits - 1.0 * false_positives
        n = max(1, len(active_violations))  # guard against empty set (step before reset)
        reward = round(max(0.01, min(0.99, raw / n)), 4)
        return reward, breakdown

    def _check_detection(self, claim_entries, all_detections, violation_id, keywords):
        """Check if any claim matches keywords AND reports the correct file."""
        expected_file = _MEDIUM_FILE_MAP.get(violation_id, "")
        for texts, fname in claim_entries:
            if is_valid_claim(texts, keywords):
                if fname and expected_file and fname != expected_file:
                    continue  # Wrong file — don't count this detection
                all_detections[violation_id] = True
                return


def _extract_claim_entries(violations_raw, report):
    """Extract per-claim entries: (texts_for_keyword_check, file_name)."""
    claim_entries = []
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
    return claim_entries