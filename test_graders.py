import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.graders.easy_grader import EasyGrader
from env.graders.medium_grader import MediumGrader
from env.graders.hard_grader import HardGrader
from env.openenv_wrapper import ReproducibilityEnvOpenEnv
from env.generators import select_violations, generate_files
import random


def test_medium_grader():
    grader = MediumGrader()

    # TEST 1: Keyword only appears in negated fix — should score 0.0
    gaming_action = json.dumps({
        "violations": [
            {
                "violation_type": "general code review finding",
                "file_name": "train.py",
                "line_number": 1,
                "suggested_fix_code": "cudnn.benchmark is not required"
            },
            {
                "violation_type": "training pipeline observation",
                "file_name": "train.py",
                "line_number": 1,
                "suggested_fix_code": "I am skipping shuffle logic"
            }
        ],
        "reproducibility_score": 0.0,
        "explanation": "Trying to trick the grader."
    })

    # TEST 2: The "Honest" Attempt (Should score > 0.0)
    honest_action = json.dumps({
        "violations": [
            {
                "violation_type": "Missing torch.backends.cudnn.benchmark = False",
                "file_name": "train.py",
                "line_number": 10,
                "suggested_fix_code": "torch.backends.cudnn.benchmark = False"
            }
        ],
        "reproducibility_score": 0.5,
        "explanation": "Properly identified the flag issue."
    })

    reward_gaming, b_g = grader.score(gaming_action)
    reward_honest, b_h = grader.score(honest_action)

    print("=== MEDIUM GRADER (all violations active) ===")
    print(f"Gaming Reward: {reward_gaming} (expected 0.0)")
    print(f"Honest Reward: {reward_honest} (expected > 0.0)")
    assert reward_gaming == 0.0, f"Gaming reward was {reward_gaming}"
    assert reward_honest > 0.0, f"Honest reward was {reward_honest}"


def test_hard_grader():
    grader = HardGrader()

    honest_action = json.dumps({
        "violations": [
            {
                "violation_type": "Missing worker seed cross file",
                "file_name": "dataset.py",
                "line_number": 5,
                "suggested_fix_code": "add worker_init_fn to DataLoader"
            }
        ],
        "reproducibility_score": 0.5,
        "explanation": "Cross file worker seed gap"
    })

    reward_honest, breakdown = grader.score(honest_action)
    print("\n=== HARD GRADER (all violations active) ===")
    print(f"Honest Reward: {reward_honest} (expected > 0.0)")
    assert reward_honest > 0.0, f"Honest reward was {reward_honest}"


def test_false_positive_penalty():
    """Agent claims violations that are NOT active — should get penalised."""
    grader = EasyGrader()

    # Only these two are active
    active = {"missing_random_seed", "missing_numpy_seed"}

    # Agent claims ALL 8 violations (6 are false positives)
    shotgun_action = json.dumps({
        "violations": [
            {"violation_type": "random.seed missing", "file_name": "train.py", "line_number": 1, "suggested_fix_code": "random.seed(42)"},
            {"violation_type": "np.random.seed missing", "file_name": "train.py", "line_number": 2, "suggested_fix_code": "np.random.seed(42)"},
            {"violation_type": "torch.manual_seed missing", "file_name": "train.py", "line_number": 3, "suggested_fix_code": "torch.manual_seed(42)"},
            {"violation_type": "cuda.manual_seed missing", "file_name": "train.py", "line_number": 4, "suggested_fix_code": "torch.cuda.manual_seed_all(42)"},
            {"violation_type": "unpinned torch", "file_name": "requirements.txt", "line_number": 1, "suggested_fix_code": "torch==2.0.0"},
            {"violation_type": "unpinned numpy", "file_name": "requirements.txt", "line_number": 2, "suggested_fix_code": "numpy==1.24.0"},
            {"violation_type": "unpinned scikit-learn", "file_name": "requirements.txt", "line_number": 3, "suggested_fix_code": "scikit-learn==1.2.2"},
            {"violation_type": "unpinned pandas", "file_name": "requirements.txt", "line_number": 4, "suggested_fix_code": "pandas==2.0.0"},
        ],
        "reproducibility_score": 0.0,
        "explanation": "Listing everything"
    })

    # Perfect agent — only claims the two active violations
    precise_action = json.dumps({
        "violations": [
            {"violation_type": "random.seed missing", "file_name": "train.py", "line_number": 1, "suggested_fix_code": "random.seed(42)"},
            {"violation_type": "np.random.seed missing", "file_name": "train.py", "line_number": 2, "suggested_fix_code": "np.random.seed(42)"},
        ],
        "reproducibility_score": 0.5,
        "explanation": "Found the two missing seeds"
    })

    reward_shotgun, _ = grader.score(shotgun_action, active)
    reward_precise, _ = grader.score(precise_action, active)

    print("\n=== FALSE POSITIVE PENALTY TEST ===")
    print(f"Shotgun (8 claims, 2 active): {reward_shotgun} (expected < 1.0)")
    print(f"Precise  (2 claims, 2 active): {reward_precise} (expected 1.0)")
    assert reward_precise > reward_shotgun, "Precise agent should beat shotgun agent"
    assert reward_precise == 1.0, f"Precise reward was {reward_precise}, expected 1.0"


def test_dynamic_violations():
    """Verify that different seeds produce different violation subsets."""
    rng1 = random.Random(42)
    rng2 = random.Random(99)

    v1 = select_violations("easy", rng1)
    v2 = select_violations("easy", rng2)

    print("\n=== DYNAMIC VIOLATION SELECTION ===")
    print(f"Seed 42 → {sorted(v1)}")
    print(f"Seed 99 → {sorted(v2)}")
    assert 5 <= len(v1) <= 9, f"Expected 5-9 violations, got {len(v1)}"
    assert 5 <= len(v2) <= 9, f"Expected 5-9 violations, got {len(v2)}"
    # Very unlikely (but not impossible) to be identical with different seeds
    print(f"Same? {v1 == v2} (expected False in most cases)")


def test_generated_files():
    """Verify that generated files reflect active violations."""
    active = {"missing_random_seed", "unpinned_torch"}
    rng = random.Random(42)
    files = generate_files("easy", active, rng)

    print("\n=== GENERATED FILES TEST ===")
    # Strip line number prefixes (e.g., "  5: code") before checking content
    import re
    def strip_lineno(line):
        return re.sub(r'^\s*\d+:\s*', '', line)

    lines = files["train.py"].split("\n")
    has_bare_random_seed = any(
        "random.seed" in l and not strip_lineno(l).strip().startswith("np.")
        for l in lines
    )
    assert not has_bare_random_seed, "random.seed should be MISSING"
    # torch should be unpinned
    req_lines = [strip_lineno(l) for l in files["requirements.txt"].split("\n")]
    assert not any("torch==" in l for l in req_lines), "torch should be UNPINNED"
    # numpy should be pinned (not in active violations)
    assert any("numpy==" in l for l in req_lines), "numpy should be PINNED"
    print("  ✅ Generated files correctly reflect active violations")


def test_openenv_wrapper_dynamic():
    """End-to-end test: wrapper returns active_violations in info."""
    from env.base_env import VIOLATION_FILE_MAP
    env = ReproducibilityEnvOpenEnv(task='easy')
    result = env.reset()

    print("\n=== OPENENV WRAPPER (dynamic) ===")
    active = result.info.get("active_violations", [])
    print(f"Active violations: {active}")
    assert len(active) >= 5, f"Expected >= 5 active violations, got {len(active)}"
    assert len(active) <= 9, f"Expected <= 9 active violations, got {len(active)}"

    # Keyword map: violation_id → (violation_type phrase, suggested_fix_code phrase)
    VIOLATION_KEYWORDS = {
        "missing_random_seed":         ("random.seed missing", "random.seed(42)"),
        "missing_numpy_seed":          ("np.random.seed missing", "np.random.seed(42)"),
        "missing_torch_seed":          ("torch.manual_seed missing", "torch.manual_seed(42)"),
        "missing_cuda_seed":           ("cuda.manual_seed missing", "torch.cuda.manual_seed_all(42)"),
        "unpinned_torch":              ("unpinned torch version", "torch==2.0.0"),
        "unpinned_numpy":              ("unpinned numpy version", "numpy==1.24.0"),
        "unpinned_scikit-learn":       ("unpinned scikit-learn version", "scikit-learn==1.2.2"),
        "unpinned_pandas":             ("unpinned pandas version", "pandas==2.0.0"),
        "missing_hashseed":            ("pythonhashseed not set", 'os.environ["PYTHONHASHSEED"] = "0"'),
        "missing_cudnn_deterministic": ("cudnn.deterministic not set", "torch.backends.cudnn.deterministic = True"),
        "missing_cudnn_benchmark_off": ("cudnn.benchmark not disabled", "torch.backends.cudnn.benchmark = False"),
    }

    # Submit an action with correct file names and keyword-rich descriptions
    violations = []
    for v in active:
        if v in VIOLATION_KEYWORDS:
            vtype, fix = VIOLATION_KEYWORDS[v]
            violations.append({
                "violation_type": vtype,
                "file_name": VIOLATION_FILE_MAP.get(v, "train.py"),
                "line_number": 1,
                "suggested_fix_code": fix,
            })

    action = json.dumps({
        "violations": violations,
        "reproducibility_score": 0.5,
        "explanation": "Identified active violations with correct files"
    })
    step_result = env.step(action)
    print(f"Step reward: {step_result.reward}")
    print(f"Breakdown: {step_result.info['score_breakdown']}")
    assert step_result.reward > 0.0, f"Expected > 0 reward for correct claims, got {step_result.reward}"


def test_file_name_validation():
    """Verify graders reject violations claimed in wrong files."""
    print("\n=== FILE NAME VALIDATION TEST ===")

    # Easy grader: random seed violation must be in train.py, not requirements.txt
    grader = EasyGrader()
    active = {"missing_random_seed"}

    # Wrong file — should NOT detect
    wrong_file_action = json.dumps({
        "violations": [
            {"violation_type": "random.seed missing", "file_name": "requirements.txt",
             "line_number": 1, "suggested_fix_code": "random.seed(42)"}
        ],
        "reproducibility_score": 0.0,
        "explanation": "Claiming seed violation in wrong file"
    })
    reward_wrong, breakdown_wrong = grader.score(wrong_file_action, active)

    # Correct file — should detect
    correct_file_action = json.dumps({
        "violations": [
            {"violation_type": "random.seed missing", "file_name": "train.py",
             "line_number": 1, "suggested_fix_code": "random.seed(42)"}
        ],
        "reproducibility_score": 0.5,
        "explanation": "Claiming seed violation in correct file"
    })
    reward_correct, breakdown_correct = grader.score(correct_file_action, active)

    print(f"  Wrong file (requirements.txt): reward={reward_wrong:.4f} (expected 0.0)")
    print(f"  Correct file (train.py):       reward={reward_correct:.4f} (expected 1.0)")
    assert reward_wrong == 0.0, f"Wrong file should score 0.0, got {reward_wrong}"
    assert reward_correct == 1.0, f"Correct file should score 1.0, got {reward_correct}"

    # Hard grader: worker_seed_cross_file must be in dataset.py, not train.py
    hard_grader = HardGrader()
    active_hard = {"worker_seed_cross_file"}

    wrong_hard = json.dumps({
        "violations": [
            {"violation_type": "worker_init_fn missing", "file_name": "train.py",
             "line_number": 1, "suggested_fix_code": "add worker_init_fn"}
        ],
        "reproducibility_score": 0.0,
        "explanation": "Wrong file for cross-file violation"
    })
    reward_wrong_h, _ = hard_grader.score(wrong_hard, active_hard)

    correct_hard = json.dumps({
        "violations": [
            {"violation_type": "worker_init_fn missing in dataset.py", "file_name": "dataset.py",
             "line_number": 1, "suggested_fix_code": "add worker_init_fn to dataset"}
        ],
        "reproducibility_score": 0.5,
        "explanation": "Correct file for cross-file violation"
    })
    reward_correct_h, _ = hard_grader.score(correct_hard, active_hard)

    print(f"  Hard wrong file (train.py):   reward={reward_wrong_h:.4f} (expected 0.0)")
    print(f"  Hard correct file (dataset.py): reward={reward_correct_h:.4f} (expected > 0)")
    assert reward_wrong_h == 0.0, f"Wrong file should score 0.0, got {reward_wrong_h}"
    assert reward_correct_h > 0.0, f"Correct file should score > 0, got {reward_correct_h}"

    print("  ✅ File name validation working correctly")


if __name__ == "__main__":
    test_medium_grader()
    test_hard_grader()
    test_false_positive_penalty()
    test_dynamic_violations()
    test_generated_files()
    test_openenv_wrapper_dynamic()
    test_file_name_validation()
    print("\n✅ All tests passed!")
