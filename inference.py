"""
inference.py — Official OpenEnv Baseline Script
================================================
Calls the live HTTP server (not local imports) to fully exercise the
OpenEnv REST contract, mirroring exactly what the eval harness does.

Required environment variables (all defined per OpenEnv hackathon spec):
  API_BASE_URL  — LLM API endpoint (OpenAI-compatible), e.g. https://api-inference.huggingface.co/v1/
  MODEL_NAME    — Model identifier, e.g. Qwen/Qwen2.5-72B-Instruct
  HF_TOKEN      — Hugging Face API key (also accepted as OPENAI_API_KEY)

Usage:
  export API_BASE_URL="https://api-inference.huggingface.co/v1/"
  export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
  export HF_TOKEN=""
  python inference.py
  python inference.py --server http://localhost:7860  # custom server URL
"""

import os
import sys
import json
import argparse
import requests
from openai import OpenAI




# ── Required environment variables (OpenEnv hackathon spec) ──────────────────
API_BASE_URL   = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# HF_TOKEN is the hackathon-standard key; OPENAI_API_KEY is accepted as fallback
API_KEY = HF_TOKEN or OPENAI_API_KEY

# Normalize API_BASE_URL: OpenAI client requires it to end in /v1 (or /v1/)
_base = API_BASE_URL.rstrip("/")
if not _base.endswith("/v1"):
    _base = _base + "/v1"
API_BASE_URL_NORMALIZED = _base + "/"

# ── Server config ─────────────────────────────────────────────────────────────
DEFAULT_SERVER  = "http://localhost:7860"
REQUEST_TIMEOUT = 30   # seconds per HTTP call
LLM_TIMEOUT     = 60   # seconds per LLM call

# ── System prompt covering all 30 violation types ────────────────────────────
SYSTEM_PROMPT = """You are an expert ML reproducibility auditor.

Given one or more ML experiment files, identify ALL reproducibility violations present in the code and propose fixes.
Respond ONLY with valid JSON. Do NOT use markdown code blocks.
Your response must exactly match this structure:
{
  "violations": [
    {
      "violation_type": "<short description of the violation>",
      "file_name": "<filename where the violation occurs>",
      "line_number": <line number as integer>,
      "suggested_fix_code": "<exact fix code snippet>"
    }
  ],
  "reproducibility_score": 0.5,
  "explanation": "<overall audit summary>"
}

CRITICAL: Only report violations ACTUALLY PRESENT in the provided code. Do NOT invent issues.

Violations to look for:
SEEDS & ENVIRONMENT:
- Missing random.seed(), np.random.seed(), torch.manual_seed(), torch.cuda.manual_seed_all()
- PYTHONHASHSEED environment variable not set
- torch.set_num_threads(1) not called
- Unpinned package versions in requirements.txt

PYTORCH DETERMINISM:
- torch.use_deterministic_algorithms(True) not set
- torch.backends.cudnn.deterministic not True
- torch.backends.cudnn.benchmark not False
- DataLoader shuffle=True without worker_init_fn or generator seed
- DataLoader missing worker_init_fn for multi-worker reproducibility
- torch.Generator() created without .manual_seed()
- np.random.default_rng() used without a seed argument
- nn.Dropout used without seed guard

CROSS-FILE & ADVANCED:
- CUBLAS_WORKSPACE_CONFIG environment variable not set
- Worker seeds not propagated across files (e.g., dataset.py)
- Incompatible package version combinations (torch vs torchvision)
- Weight initialisation without seed guard (nn.init without prior seed)
- CLI args overriding config seeds without re-seeding all libraries
- multiprocessing workers spawned without seed propagation
- PYTHONHASHSEED set too late (after Python startup, inside main())
- torch.use_deterministic_algorithms called without CUBLAS_WORKSPACE_CONFIG set first
- torch.cuda.manual_seed() used instead of manual_seed_all() (misses multi-GPU)
"""


def _call_server(server: str, method: str, path: str, payload: dict | None = None) -> dict:
    """Make a request to the OpenEnv server."""
    url = f"{server.rstrip('/')}{path}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        else:
            resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print(f"  ❌ Cannot connect to server at {server}.")
        print(f"     Start the server first:  uvicorn server:app --host 0.0.0.0 --port 7860")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"  ❌ Server request timed out ({REQUEST_TIMEOUT}s)")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"  ❌ Server error {e.response.status_code}: {e.response.text[:200]}")
        raise


def evaluate_task(task_name: str, client: OpenAI, server: str) -> float:
    """Run one task episode via the HTTP server and return the reward."""
    print(f"\n🔍 Evaluating Task: {task_name.upper()}")

    # 1. Reset via server
    reset_data = _call_server(server, "POST", "/reset", {"task": task_name})
    observation = reset_data.get("observation", "")
    active = reset_data.get("info", {}).get("active_violations", [])
    print(f"   Active violations this episode: {len(active)}")

    # 2. Call LLM via OpenAI client (required by OpenEnv hackathon spec)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": observation},
            ],
            temperature=0.1,
            timeout=LLM_TIMEOUT,
        )
        action_str = response.choices[0].message.content.strip()

        # Strip accidental markdown fences
        if action_str.startswith("```json"):
            action_str = action_str[7:]
        if action_str.startswith("```"):
            action_str = action_str[3:]
        if action_str.endswith("```"):
            action_str = action_str[:-3]
        action_str = action_str.strip()

        # Validate JSON before sending
        action_dict = json.loads(action_str)

    except json.JSONDecodeError as e:
        print(f"   ⚠️  LLM returned invalid JSON: {e}")
        action_dict = {"violations": [], "reproducibility_score": 0.0, "explanation": "Parse error"}
    except Exception as e:
        print(f"   ⚠️  LLM call failed: {e}")
        return 0.0

    # 3. Step via server
    step_data = _call_server(server, "POST", "/step", {"task": task_name, "action": action_dict})

    reward = step_data.get("reward", 0.0)
    breakdown = step_data.get("info", {}).get("score_breakdown", {})

    for check, passed in breakdown.items():
        icon = "✅" if passed else "❌"
        print(f"    {icon}  {check}")
    print(f"    ── Reward: {reward:.4f}")

    return reward


def main():
    parser = argparse.ArgumentParser(
        description="OpenEnv Reproducibility Auditor — Baseline Inference Script"
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"OpenEnv server URL (default: {DEFAULT_SERVER})",
    )
    args = parser.parse_args()

    # ── Validate required environment variables ───────────────────────────────
    missing = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not API_KEY:
        missing.append("HF_TOKEN (or OPENAI_API_KEY)")

    if missing:
        print("❌ ERROR: Missing required environment variables:")
        for var in missing:
            print(f"   export {var}='...'")
        print("\nAll three variables are required by the OpenEnv hackathon spec.")
        sys.exit(1)

    print(f"\n{'='*56}")
    print(f"  OpenEnv Reproducibility Auditor — Baseline")
    print(f"{'='*56}")
    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print(f"  HF_TOKEN     : {'✅ set' if HF_TOKEN else '❌ not set'}")
    print(f"  Server       : {args.server}")
    print(f"{'='*56}\n")

    # ── Verify server is reachable (automated ping gate) ─────────────────────
    print(f"📡  Pinging server: {args.server}/health ...")
    health = _call_server(args.server, "GET", "/health")
    status = health.get("status", "unknown")
    tasks_loaded = health.get("tasks_loaded", [])
    print(f"    Status: {status} | Tasks loaded: {tasks_loaded}")

    if status != "ok":
        print("❌ Server health check failed. Aborting.")
        sys.exit(1)

    # ── Build OpenAI client (spec: must use OpenAI client for all LLM calls) ─
    client = OpenAI(base_url=API_BASE_URL_NORMALIZED, api_key=API_KEY)

    # ── Run all three tasks ───────────────────────────────────────────────────
    tasks = ["easy", "medium", "hard"]
    scores: dict[str, float] = {}

    print("🚀 Starting inference run ...\n")
    for task in tasks:
        scores[task] = evaluate_task(task, client, args.server)

    # ── Summary ───────────────────────────────────────────────────────────────
    final_avg = sum(scores.values()) / len(scores)

    print(f"\n{'='*56}")
    print(f"  📊 PER-TASK SCORES:")
    for task, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"     {task:8s}: {score:.4f}  {bar}")
    print(f"  {'─'*52}")
    print(f"  🏁 FINAL AVERAGE SCORE: {final_avg:.4f}")
    print(f"{'='*56}\n")

    # Exit 0 = success (required by baseline-reproduces gate)
    sys.exit(0)


if __name__ == "__main__":
    main()