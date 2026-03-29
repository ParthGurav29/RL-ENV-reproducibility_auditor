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
    # random.seed should NOT appear (violation is active = seed is missing)
    # But np.random.seed may appear — use line-level check
    lines = files["train.py"].split("\n")
    has_bare_random_seed = any(
        "random.seed" in l and not l.strip().startswith("np.") for l in lines
    )
    assert not has_bare_random_seed, "random.seed should be MISSING"
    # torch should be unpinned
    assert "torch==" not in files["requirements.txt"], "torch should be UNPINNED"
    # numpy should be pinned (not in active violations)
    assert "numpy==" in files["requirements.txt"], "numpy should be PINNED"
    print("  ✅ Generated files correctly reflect active violations")


def test_openenv_wrapper_dynamic():
    """End-to-end test: wrapper returns active_violations in info."""
    env = ReproducibilityEnvOpenEnv(task='easy')
    result = env.reset()

    print("\n=== OPENENV WRAPPER (dynamic) ===")
    active = result.info.get("active_violations", [])
    print(f"Active violations: {active}")
    assert len(active) >= 5, f"Expected >= 5 active violations, got {len(active)}"
    assert len(active) <= 9, f"Expected <= 9 active violations, got {len(active)}"

    # Submit an action that claims only the active ones
    action = json.dumps({
        "violations": [
            {"violation_type": v, "file_name": "train.py", "line_number": 1, "suggested_fix_code": v}
            for v in active
        ],
        "reproducibility_score": 0.5,
        "explanation": "Only claiming active violations"
    })
    step_result = env.step(action)
    print(f"Step reward: {step_result.reward}")
    print(f"Breakdown: {step_result.info['score_breakdown']}")


if __name__ == "__main__":
    test_medium_grader()
    test_hard_grader()
    test_false_positive_penalty()
    test_dynamic_violations()
    test_generated_files()
    test_openenv_wrapper_dynamic()
    print("\n✅ All tests passed!")
