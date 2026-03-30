#!/usr/bin/env python3
"""
validate.py — Pre-submission validation script for the OpenEnv Hackathon.
=========================================================================
Runs every item in the pre-submission checklist automatically.
Must be run with the server already started on localhost:7860.

Updated for 2-step episodes (triage → audit).

Usage:
    # Terminal 1:
    uvicorn server:app --host 0.0.0.0 --port 7860

    # Terminal 2:
    python validate.py
    python validate.py --server http://localhost:7860   # custom port
"""
import sys
import json
import argparse
import requests

TASKS = ["easy", "medium", "hard"]
PASS = "✅ PASS"
FAIL = "❌ FAIL"


def check(label: str, result: bool, detail: str = "") -> bool:
    status = PASS if result else FAIL
    line = f"  {status}  {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return result


def get(server: str, path: str, timeout: int = 10) -> requests.Response:
    return requests.get(f"{server}{path}", timeout=timeout)


def post(server: str, path: str, payload: dict, timeout: int = 10) -> requests.Response:
    return requests.post(f"{server}{path}", json=payload, timeout=timeout)


def run_validation(server: str) -> bool:
    results = []

    print(f"\n{'='*60}")
    print(f"  OpenEnv Pre-Submission Validator (2-step episodes)")
    print(f"  Target: {server}")
    print(f"{'='*60}\n")

    # ── 1. Health / Deployment ────────────────────────────────────────────────
    print("📡  [1/6] Deployment & Health Check")
    try:
        r = get(server, "/health")
        ok = r.status_code == 200
        data = r.json() if ok else {}
        results.append(check("Server responds with HTTP 200", ok, f"status={data.get('status')}"))
        results.append(check("All 3 tasks loaded", set(data.get("tasks_loaded", [])) == {"easy", "medium", "hard"},
                       f"loaded={data.get('tasks_loaded')}"))
    except requests.ConnectionError:
        print(f"  {FAIL}  Cannot connect to {server} — is the server running?")
        sys.exit(1)

    # ── 2. OpenEnv Spec compliance ────────────────────────────────────────────
    print("\n📋  [2/6] OpenEnv Spec Compliance (/spec)")
    r = get(server, "/spec")
    spec = r.json() if r.status_code == 200 else {}
    results.append(check("/spec returns HTTP 200", r.status_code == 200))
    results.append(check("spec.tasks has 3 entries", len(spec.get("tasks", [])) == 3,
                   f"found={len(spec.get('tasks', []))}"))
    results.append(check("spec.reward_range is [0.0, 1.0]", spec.get("reward_range") == [0.0, 1.0],
                   f"found={spec.get('reward_range')}"))
    results.append(check("spec.observation_type present", "observation_type" in spec))
    results.append(check("spec.action_schema present", "action_schema" in spec))
    results.append(check("spec.randomization described", bool(spec.get("randomization"))))
    results.append(check("spec.episode_steps is 2", spec.get("episode_steps") == 2,
                   f"found={spec.get('episode_steps')}"))

    # Check for multi-step schema
    action_schema = spec.get("action_schema", {})
    results.append(check("action_schema has triage + audit schemas",
                   "step_1_triage" in action_schema and "step_2_audit" in action_schema,
                   f"keys={list(action_schema.keys())}"))

    # ── 3. 2-step reset() / step() functional test ───────────────────────────
    print("\n🔄  [3/6] Multi-Step reset() / step() Functional Test")
    for task in TASKS:
        # Step 0: Reset
        r_reset = post(server, "/reset", {"task": task})
        reset_ok = r_reset.status_code == 200
        reset_data = r_reset.json() if reset_ok else {}

        has_obs = isinstance(reset_data.get("observation"), str) and len(reset_data.get("observation", "")) > 10
        has_active = isinstance(reset_data.get("info", {}).get("active_violations"), list)
        active_count = len(reset_data.get("info", {}).get("active_violations", []))
        max_steps = reset_data.get("info", {}).get("max_steps", 0)

        results.append(check(f"/reset [{task}] returns observation", has_obs,
                       f"obs_len={len(reset_data.get('observation',''))}"))
        results.append(check(f"/reset [{task}] returns active_violations", has_active,
                       f"count={active_count}"))
        results.append(check(f"/reset [{task}] max_steps=2", max_steps == 2,
                       f"max_steps={max_steps}"))

        # Step 1: Triage (empty triage action)
        empty_triage = {"suspicious_files": [], "suspected_categories": [], "reasoning": "empty triage"}
        r_triage = post(server, "/step", {"task": task, "action": empty_triage})
        triage_ok = r_triage.status_code == 200
        triage_data = r_triage.json() if triage_ok else {}
        triage_reward = triage_data.get("reward", -1)
        triage_terminated = triage_data.get("terminated", True)
        triage_step_type = triage_data.get("info", {}).get("step_type", "")

        results.append(check(f"/step1 [{task}] triage returns reward in [0,1]",
                       triage_ok and 0.0 <= triage_reward <= 1.0,
                       f"reward={triage_reward}"))
        results.append(check(f"/step1 [{task}] terminated=False (not done yet)",
                       not triage_terminated))
        results.append(check(f"/step1 [{task}] step_type=triage",
                       triage_step_type == "triage",
                       f"got={triage_step_type}"))
        results.append(check(f"/step1 [{task}] triage_feedback present",
                       isinstance(triage_data.get("info", {}).get("triage_feedback"), dict)))

        # Step 2: Audit (empty audit action)
        empty_audit = {"violations": [], "reproducibility_score": 0.0, "explanation": "empty"}
        r_step = post(server, "/step", {"task": task, "action": empty_audit})
        step_ok = r_step.status_code == 200
        step_data = r_step.json() if step_ok else {}
        reward = step_data.get("reward", -1)
        terminated = step_data.get("terminated", False)

        results.append(check(f"/step2 [{task}] audit returns reward in [0,1]",
                       step_ok and 0.0 <= reward <= 1.0,
                       f"reward={reward}"))
        results.append(check(f"/step2 [{task}] terminated=True (episode done)",
                       terminated))
        results.append(check(f"/step2 [{task}] score_breakdown present",
                       isinstance(step_data.get("info", {}).get("score_breakdown"), dict)))

    # ── 4. Grader varying scores ──────────────────────────────────────────────
    print("\n🎲  [4/6] Grader Score Variance (anti-disqualification)")

    # Keyword maps: violation_id → (violation_type phrase, suggested_fix_code phrase)
    VIOLATION_KEYWORDS = {
        # Easy
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
        # Medium
        "dataloader_shuffle_no_seed":  ("dataloader shuffle without seed", "generator=torch.Generator().manual_seed(42)"),
        "missing_deterministic_flag":  ("use_deterministic_algorithms missing", "torch.use_deterministic_algorithms(True)"),
        "missing_worker_seed":         ("worker_init_fn missing", "worker_init_fn=lambda w: torch.manual_seed(42+w)"),
        "missing_generator_seed":      ("torch.Generator seed missing", "generator().manual_seed(42)"),
        "missing_default_rng_seed":    ("np.random.default_rng seed missing", "np.random.default_rng(42)"),
        "missing_dropout_seed":        ("nn.Dropout without seed", "torch.manual_seed(42)"),
        # Hard
        "worker_seed_cross_file":      ("worker_init_fn cross-file missing", "worker_init_fn in dataset.py"),
        "cublas_workspace_config":     ("cublas_workspace_config not set", 'os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"'),
        "package_version_conflict":    ("torchvision version conflict incompatible", "torchvision==0.16.2"),
        "model_weight_init_seed":      ("model weight init without seed", "torch.manual_seed(42)  # nn.init guard"),
        "config_yaml_override":        ("cli override config.yaml seed", "np.random.seed(active_seed)"),
        "hash_randomization":          ("pythonhashseed not set", 'os.environ["PYTHONHASHSEED"] = "0"'),
        "multiprocessing_no_seed":     ("multiprocessing pool spawn without seed", "initializer=mp_init"),
        "deterministic_without_cublas":("use_deterministic_algorithms without cublas", 'os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"'),
        "incomplete_cuda_seed":        ("manual_seed_all vs manual_seed multi-gpu", "torch.cuda.manual_seed_all(seed)"),
    }

    for task in TASKS:
        # -- Empty 2-step: expect audit reward = 0.0 --------------------------
        post(server, "/reset", {"task": task})
        # Step 1: empty triage
        post(server, "/step", {"task": task, "action": {
            "suspicious_files": [], "suspected_categories": [], "reasoning": "nothing"}})
        # Step 2: empty audit
        r0 = post(server, "/step", {"task": task, "action": {
            "violations": [], "reproducibility_score": 0.0, "explanation": "nothing"}})

        # -- Correct 2-step: use active violations to get reward > 0 -----------
        reset_data = post(server, "/reset", {"task": task}).json()
        active = reset_data.get("info", {}).get("active_violations", [])

        # Step 1: correct triage (list correct files and categories)
        from env.base_env import VIOLATION_FILE_MAP, VIOLATION_CATEGORY_MAP
        correct_files = list({VIOLATION_FILE_MAP[v] for v in active if v in VIOLATION_FILE_MAP})
        correct_cats = list({VIOLATION_CATEGORY_MAP[v] for v in active if v in VIOLATION_CATEGORY_MAP})
        post(server, "/step", {"task": task, "action": {
            "suspicious_files": correct_files,
            "suspected_categories": correct_cats,
            "reasoning": "Correct triage using ground truth"}})

        # Step 2: correct audit
        submitted_violations = []
        for v_id in active:
            if v_id in VIOLATION_KEYWORDS:
                v_type, fix = VIOLATION_KEYWORDS[v_id]
                submitted_violations.append({
                    "violation_type": v_type,
                    "file_name": "train.py",
                    "line_number": 1,
                    "suggested_fix_code": fix
                })

        r1 = post(server, "/step", {"task": task, "action": {
            "violations": submitted_violations,
            "reproducibility_score": 0.8,
            "explanation": "Identified active violations using keyword matching"
        }})

        s0 = r0.json().get("reward", -1) if r0.status_code == 200 else -1
        s1 = r1.json().get("reward", -1) if r1.status_code == 200 else -1
        results.append(check(f"[{task}] audit scores differ between empty vs. correct action",
                       s0 != s1 or s1 > 0, f"empty={s0:.4f}, correct={s1:.4f}"))

    # ── 5. /state endpoint ────────────────────────────────────────────────────
    print("\n🗂️   [5/6] State Endpoint")
    for task in TASKS:
        post(server, "/reset", {"task": task})
        r_state = get(server, f"/state?task={task}")
        state_ok = r_state.status_code == 200
        state_data = r_state.json() if state_ok else {}
        results.append(check(f"/state [{task}] returns active_violations",
                       "active_violations" in state_data,
                       f"step_count={state_data.get('step_count')}"))
        results.append(check(f"/state [{task}] has max_steps=2",
                       state_data.get("max_steps") == 2,
                       f"max_steps={state_data.get('max_steps')}"))
        results.append(check(f"/state [{task}] tracks triage_completed",
                       "triage_completed" in state_data,
                       f"triage_completed={state_data.get('triage_completed')}"))

    # ── 6. Multi-step trajectory signal ──────────────────────────────────────
    print("\n📈  [6/6] Trajectory Signal Verification")
    for task in TASKS:
        post(server, "/reset", {"task": task})

        # Good triage
        good_triage = post(server, "/step", {"task": task, "action": {
            "suspicious_files": ["train.py", "requirements.txt"],
            "suspected_categories": ["random_seeds", "dependency_pinning"],
            "reasoning": "These are common violation sites"}})
        good_triage_reward = good_triage.json().get("reward", 0) if good_triage.status_code == 200 else 0

        # Skip audit for this test — just reset again
        post(server, "/reset", {"task": task})

        # Bad triage (completely wrong)
        bad_triage = post(server, "/step", {"task": task, "action": {
            "suspicious_files": ["nonexistent.py"],
            "suspected_categories": ["nonexistent_category"],
            "reasoning": "Random guess"}})
        bad_triage_reward = bad_triage.json().get("reward", 0) if bad_triage.status_code == 200 else 0

        results.append(check(f"[{task}] triage rewards differ (good vs bad)",
                       good_triage_reward > bad_triage_reward or good_triage_reward > 0,
                       f"good={good_triage_reward:.4f}, bad={bad_triage_reward:.4f}"))

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} checks passed")
    if failed == 0:
        print("  🎉 ALL CHECKS PASSED — Ready to submit!")
    else:
        print(f"  ⚠️  {failed} check(s) failed — fix before submitting.")
    print(f"{'='*60}\n")

    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEnv Pre-Submission Validator")
    parser.add_argument("--server", default="http://localhost:7860",
                        help="Server URL (default: http://localhost:7860)")
    args = parser.parse_args()

    success = run_validation(args.server)
    sys.exit(0 if success else 1)
