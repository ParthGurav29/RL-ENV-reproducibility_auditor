"""
Hard Grader — multi-file experiment with subtle, cross-file reproducibility gaps.
Keywords checked only in violation_type and suggested_fix_code fields.
Bidirectional negation guard (25 chars before + after keyword).
file_name validated against expected file for each violation type.
"""

import json

from env.graders.utils import is_valid_claim

HARD_CHECKS = {
    "worker_seed_cross_file": "worker_init_fn in dataset.py needed",
    "cublas_workspace_config": "CUBLAS_WORKSPACE_CONFIG env var not set",
    "package_version_conflict": "torch and torchvision versions are incompatible",
    "model_weight_init_seed": "model.py weight init without seed guard",
    "config_yaml_override": "CLI arg overrides config seed without re-seeding all libraries",
    "hash_randomization": "PYTHONHASHSEED not set",
    "multiprocessing_no_seed": "multiprocessing workers spawned without seed propagation",
}

ALL_HARD_CHECKS = list(HARD_CHECKS.keys())

# Local file map for validation (avoids circular import with base_env)
_HARD_FILE_MAP = {
    "worker_seed_cross_file": "dataset.py",
    "cublas_workspace_config": "train.py",
    "package_version_conflict": "requirements.txt",
    "model_weight_init_seed": "model.py",
    "config_yaml_override": "train.py",
    "hash_randomization": "train.py",
    "multiprocessing_no_seed": "train.py",
}


class HardGrader:
    def score(self, action_json: str, active_violations: set[str] | None = None) -> tuple[float, dict]:
        if active_violations is None:
            active_violations = set(ALL_HARD_CHECKS)

        all_detections = {k: False for k in ALL_HARD_CHECKS}

        try:
            report = json.loads(action_json)
            violations_raw = report.get("violations", [])

            # Build per-claim entries with file_name for validation
            claim_entries = []
            files_mentioned = set()

            if violations_raw and isinstance(violations_raw[0], dict):
                for v in violations_raw:
                    fname = str(v.get("file_name", "")).lower()
                    vtype = str(v.get("violation_type", "")).lower()
                    sfix = str(v.get("suggested_fix_code", "")).lower()
                    texts = [t for t in [vtype, sfix] if t]
                    if texts:
                        claim_entries.append((texts, fname))
                    if fname:
                        files_mentioned.add(fname)
            else:
                flat_texts = [str(v).lower() for v in violations_raw] + \
                              [str(f).lower() for f in report.get("fixes", [])]
                claim_entries = [(flat_texts, "")]
                for t in flat_texts:
                    if "dataset" in t:
                        files_mentioned.add("dataset.py")
        except (json.JSONDecodeError, AttributeError):
            breakdown = {k: False for k in ALL_HARD_CHECKS if k in active_violations}
            return 0.0, breakdown

        # Cross-file check: broadened to match "dataset" in file names or claim texts
        has_dataset = (
            any("dataset" in f for f in files_mentioned)
            or any("dataset" in t for texts, _ in claim_entries for t in texts)
        )

        # worker_seed_cross_file: needs dataset context + worker keywords
        if has_dataset:
            self._check_detection(claim_entries, all_detections, "worker_seed_cross_file", [
                "worker_init_fn", "worker seed", "worker init", "worker_init",
                "dataloader worker", "cross-file", "cross file", "crossfile",
                "seed not propagated", "seed propagation", "propagate seed",
                "seed across", "dataset seed", "seed_worker", "worker seeding",
                "worker id seed", "worker process seed",
            ])
        # Also detect with very specific keywords even without file context
        if not all_detections["worker_seed_cross_file"]:
            self._check_detection(claim_entries, all_detections, "worker_seed_cross_file", [
                "cross-file seed", "cross file seed", "worker_init_fn in dataset",
                "dataset.py worker", "dataset worker_init", "dataset worker_init_fn",
                "worker_init_fn missing in dataset", "seed_worker missing",
            ])

        self._check_detection(claim_entries, all_detections, "cublas_workspace_config", [
            "cublas_workspace_config", "cublas", "workspace_config",
            "cublas workspace", "cublas_workspace",
        ])
        self._check_detection(claim_entries, all_detections, "package_version_conflict", [
            "torchvision", "version conflict", "incompatible",
            "0.14", "0.15", "0.17", "version mismatch",
            "conflicting versions", "version incompatibility", "incompatible versions",
            "package version conflict", "version compatibility",
            "torch and torchvision", "torchvision version",
        ])
        self._check_detection(claim_entries, all_detections, "model_weight_init_seed", [
            "weight init", "nn.init", "model weight", "initialisation seed",
            "initialization seed", "init seed", "xavier", "kaiming",
            "weight initialization", "weight initialisation",
            "model.py", "weight_init", "init_weights",
        ])
        self._check_detection(claim_entries, all_detections, "config_yaml_override", [
            "config.yaml", "config yaml", "cli override", "cli arg",
            "command line", "--seed", "--override", "override seed",
            "seed override", "numpy not re-seeded", "partial re-seed",
            "re-seed", "reseed", "config seed", "not re-seeded",
            "libraries not seeded", "all libraries", "re-seeding",
            "argparse", "args.seed", "argument seed", "seed arg",
            "seed not propagated", "seed mismatch", "inconsistent seed",
            "seed not applied", "parse_args", "overrid",
            "np.random.seed(active", "np.random.seed(seed",
        ])
        self._check_detection(claim_entries, all_detections, "hash_randomization", [
            "pythonhashseed", "hash seed", "hash randomization",
            "set ordering", "python hash seed", "hashseed",
        ])
        self._check_detection(claim_entries, all_detections, "multiprocessing_no_seed", [
            "multiprocessing", "pool(", "pool.map", "worker process",
            "mp.pool", "process pool", "mp.process",
        ])

        breakdown = {k: all_detections[k] for k in ALL_HARD_CHECKS if k in active_violations}
        hits = sum(1 for v in breakdown.values() if v)
        false_positives = sum(1 for k in ALL_HARD_CHECKS if k not in active_violations and all_detections[k])
        raw = hits - 1.0 * false_positives
        n = max(1, len(active_violations))  # guard against empty set (step before reset)
        reward = round(max(0.0, min(1.0, raw / n)), 4)
        return reward, breakdown

    def _check_detection(self, claim_entries, all_detections, violation_id, keywords):
        """Check if any claim matches keywords AND reports the correct file."""
        expected_file = _HARD_FILE_MAP.get(violation_id, "")
        for texts, fname in claim_entries:
            if is_valid_claim(texts, keywords):
                if fname and expected_file and fname != expected_file:
                    continue  # Wrong file — don't count this detection
                all_detections[violation_id] = True
                return