"""
inference.py — Official OpenEnv Baseline Script
================================================
Calls the live HTTP server (not local imports) to fully exercise the
OpenEnv REST contract, mirroring exactly what the eval harness does.

Multi-step episodes (2-step):
  Step 1 — Triage: Identify suspicious files and violation categories
  Step 2 — Audit:  Full violation report using triage feedback

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
import time
import argparse
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file so API_BASE_URL, MODEL_NAME, HF_TOKEN are available
load_dotenv()

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
LLM_MAX_RETRIES = 3    # retry LLM calls on failure
LLM_RETRY_DELAY = 2    # seconds between retries

# ── System prompts ───────────────────────────────────────────────────────────

TRIAGE_PROMPT = """You are an expert ML reproducibility auditor performing an initial triage.

Given ML experiment files, identify which FILES are most likely to contain reproducibility violations,
and which CATEGORIES of violations you suspect are present.

Available violation categories:
- random_seeds: Missing random, numpy, torch, CUDA seed calls
- dependency_pinning: Unpinned package versions, version conflicts
- determinism_flags: Missing cudnn.deterministic, cudnn.benchmark, use_deterministic_algorithms
- environment_config: Missing PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG, set_num_threads
- dataloader_reproducibility: DataLoader shuffle without seed, missing worker_init_fn, generator seed
- model_initialization: Weight init without seed guard, Dropout without seed
- configuration_management: CLI args overriding config seeds without re-seeding
- multiprocessing: Workers spawned without seed propagation
- rng_initialization: np.random.default_rng() without seed

Respond ONLY with valid JSON. Do NOT use markdown code blocks.
Your response must exactly match this structure:
{
  "suspicious_files": ["train.py", "requirements.txt"],
  "suspected_categories": ["random_seeds", "dependency_pinning"],
  "reasoning": "Brief explanation of why these files and categories are suspicious"
}

CRITICAL: Only flag files and categories that genuinely appear suspicious from the code. Do NOT list everything — precision matters as much as recall.
"""

AUDIT_PROMPT = """You are an expert ML reproducibility auditor.

Given ML experiment files AND feedback from your preliminary triage, identify ALL reproducibility violations present in the code and propose fixes.

Use the triage feedback to focus your analysis — confirmed files and categories are more likely to contain violations.

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

IMPORTANT RULES:
1. Report EVERY violation as a SEPARATE entry — do NOT merge related issues.
2. Be thorough. Check EVERY item in the checklist below against the code.
3. ONLY report violations that ACTUALLY EXIST in the code. Each false positive costs as much as a missed detection.
4. Use the EXACT violation_type descriptions shown below when reporting.
5. Verify each violation by looking at the actual code — don't guess.

CHECK EACH OF THESE SYSTEMATICALLY (use the EXACT violation_type phrase shown):

SEEDS (check train.py for each one individually):
- "random.seed missing" → random.seed() not called → fix: random.seed(42)
- "np.random.seed missing" → np.random.seed() not called → fix: np.random.seed(42)
- "torch.manual_seed missing" → torch.manual_seed() not called → fix: torch.manual_seed(42)
- "torch.cuda.manual_seed missing" → torch.cuda.manual_seed_all() not called → fix: torch.cuda.manual_seed_all(42)

ENVIRONMENT & DETERMINISM (check train.py):
- "PYTHONHASHSEED not set" → os.environ["PYTHONHASHSEED"] missing → fix: os.environ["PYTHONHASHSEED"] = "0"
- "cudnn.deterministic not set" → torch.backends.cudnn.deterministic not True → fix: torch.backends.cudnn.deterministic = True
- "cudnn.benchmark not disabled" → torch.backends.cudnn.benchmark not False → fix: torch.backends.cudnn.benchmark = False

DEPENDENCIES (check requirements.txt):
- "unpinned torch version" → torch without ==version → fix: torch==2.0.0
- "unpinned numpy version" → numpy without ==version → fix: numpy==1.24.0
- "unpinned scikit-learn version" → scikit-learn without ==version → fix: scikit-learn==1.2.2
- "unpinned pandas version" → pandas without ==version → fix: pandas==2.0.0
- "torchvision version conflict" → incompatible torch/torchvision → fix: torchvision==0.16.2

PYTORCH DETERMINISM:
- "use_deterministic_algorithms missing" → torch.use_deterministic_algorithms(True) not called → fix: torch.use_deterministic_algorithms(True)
- "DataLoader shuffle without seed" → shuffle=True but no generator seed → fix: generator=torch.Generator().manual_seed(42)
- "worker_init_fn missing" → DataLoader without worker_init_fn → fix: worker_init_fn=lambda w: torch.manual_seed(42+w)
- "Generator seed missing" → torch.Generator() without .manual_seed() → fix: g.manual_seed(42)
- "default_rng seed missing" → np.random.default_rng() without seed → fix: np.random.default_rng(42)
- "Dropout without seed guard" → nn.Dropout without seed → fix: torch.manual_seed(42) before dropout

CROSS-FILE & ADVANCED:
- "CUBLAS_WORKSPACE_CONFIG not set" → missing env var → fix: os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
- "worker_init_fn cross-file missing" → dataset.py worker seed not propagated → fix: add worker_init_fn function to dataset.py
- "weight init without seed guard" → nn.init without torch.manual_seed() → fix: torch.manual_seed(42) before init
- "config yaml seed override" → CLI args override config seed without re-seeding → fix: re-seed all libraries in main()
- "multiprocessing pool without seed" → workers spawned without seed → fix: Pool(initializer=mp_init)

PHRASING HINTS — these phrases MUST appear in your violation_type to be detected:
- For cuda seed: include "torch.cuda.manual_seed" in violation_type (e.g., "torch.cuda.manual_seed missing")
- For dropout: include "dropout" AND "seed" (e.g., "Dropout without seed guard")
- For torchvision conflict: include "torchvision" or "version conflict" in violation_type
- For cross-file worker: include "worker_init_fn" AND mention "dataset.py" in file_name
- For deterministic flag: include "use_deterministic_algorithms" or "deterministic algorithms" in violation_type
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


def _call_llm_with_retry(client: OpenAI, messages: list, model: str) -> str:
    """Call the LLM with retry logic. Returns the response text."""
    last_error = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                timeout=LLM_TIMEOUT,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            if attempt < LLM_MAX_RETRIES:
                print(f"   ⚠️  LLM call failed (attempt {attempt}/{LLM_MAX_RETRIES}): {e}")
                time.sleep(LLM_RETRY_DELAY)
    raise last_error if last_error else RuntimeError("LLM call failed")


def _parse_llm_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON from LLM response."""
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def _sanitize_audit_dict(audit: dict) -> dict:
    """Fix common LLM typos in violation keys before sending to server.

    The LLM occasionally misspells 'violation_type' as 'violution_type'
    or similar. Pydantic rejects these and crashes the run.
    """
    KEY_FIXES = {
        "violution_type": "violation_type",
        "violtion_type": "violation_type",
        "violationtype": "violation_type",
        "violation_tpye": "violation_type",
        "violtion": "violation_type",
        "file_nmae": "file_name",
        "file_nme": "file_name",
        "filen_ame": "file_name",
        "line_num": "line_number",
        "line_numbr": "line_number",
        "linenumber": "line_number",
        "suggested_fix": "suggested_fix_code",
        "suggested_fix_cod": "suggested_fix_code",
        "fix_code": "suggested_fix_code",
    }
    violations = audit.get("violations", [])
    if not violations:
        return audit
    cleaned = []
    for v in violations:
        if not isinstance(v, dict):
            cleaned.append(v)
            continue
        fixed = {}
        for key, val in v.items():
            fixed_key = KEY_FIXES.get(key, key)
            fixed[fixed_key] = val
        cleaned.append(fixed)
    audit["violations"] = cleaned
    return audit


def evaluate_task(task_name: str, client: OpenAI, server: str) -> tuple[float, float]:
    """Run one 2-step task episode via the HTTP server. Returns (triage_reward, audit_reward)."""
    print(f"\n🔍 Evaluating Task: {task_name.upper()}")

    # ── Step 0: Reset ─────────────────────────────────────────────────────────
    reset_data = _call_server(server, "POST", "/reset", {"task": task_name})
    observation = reset_data.get("observation", "")
    active = reset_data.get("info", {}).get("active_violations", [])
    max_steps = reset_data.get("info", {}).get("max_steps", 2)
    print(f"   Active violations this episode: {len(active)}")
    print(f"   Episode steps: {max_steps}")

    # ── Step 1: Triage ────────────────────────────────────────────────────────
    print(f"\n   📋 Step 1/2: Triage ...")
    try:
        triage_str = _call_llm_with_retry(client, [
            {"role": "system", "content": TRIAGE_PROMPT},
            {"role": "user",   "content": observation},
        ], MODEL_NAME)
        triage_dict = _parse_llm_json(triage_str)
    except json.JSONDecodeError as e:
        print(f"   ⚠️  LLM returned invalid triage JSON: {e}")
        triage_dict = {"suspicious_files": [], "suspected_categories": [], "reasoning": "Parse error"}
    except Exception as e:
        print(f"   ⚠️  LLM triage call failed after {LLM_MAX_RETRIES} attempts: {e}")
        triage_dict = {"suspicious_files": [], "suspected_categories": [], "reasoning": str(e)}

    # Submit triage to server
    triage_result = _call_server(server, "POST", "/step", {"task": task_name, "action": triage_dict})
    triage_reward = triage_result.get("reward", 0.0)
    triage_feedback = triage_result.get("info", {}).get("triage_feedback", {})
    enhanced_obs = triage_result.get("observation", observation)

    print(f"   📊 Triage reward: {triage_reward:.4f}")
    if triage_feedback.get("files_confirmed"):
        print(f"   ✅ Confirmed files: {', '.join(triage_feedback['files_confirmed'])}")
    if triage_feedback.get("categories_confirmed"):
        print(f"   ✅ Confirmed categories: {', '.join(triage_feedback['categories_confirmed'])}")

    # ── Step 2: Full Audit ────────────────────────────────────────────────────
    print(f"\n   🔬 Step 2/2: Full Audit ...")
    try:
        audit_str = _call_llm_with_retry(client, [
            {"role": "system", "content": AUDIT_PROMPT},
            {"role": "user",   "content": enhanced_obs},
        ], MODEL_NAME)
        audit_dict = _parse_llm_json(audit_str)
    except json.JSONDecodeError as e:
        print(f"   ⚠️  LLM returned invalid audit JSON: {e}")
        audit_dict = {"violations": [], "reproducibility_score": 0.0, "explanation": "Parse error"}
    except Exception as e:
        print(f"   ⚠️  LLM audit call failed after {LLM_MAX_RETRIES} attempts: {e}")
        return triage_reward, 0.0

    # Sanitize common LLM typos in violation keys (e.g. violution_type)
    audit_dict = _sanitize_audit_dict(audit_dict)

    # Submit audit to server
    step_data = _call_server(server, "POST", "/step", {"task": task_name, "action": audit_dict})

    audit_reward = step_data.get("reward", 0.0)
    breakdown = step_data.get("info", {}).get("score_breakdown", {})

    for check, passed in breakdown.items():
        icon = "✅" if passed else "❌"
        print(f"    {icon}  {check}")
    print(f"    ── Triage reward:  {triage_reward:.4f}")
    print(f"    ── Audit reward:   {audit_reward:.4f}")

    return triage_reward, audit_reward


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
    print(f"  Episode mode : 2-step (triage → audit)")
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
    triage_scores: dict[str, float] = {}
    audit_scores: dict[str, float] = {}

    print("🚀 Starting inference run ...\n")
    for task in tasks:
        t_score, a_score = evaluate_task(task, client, args.server)
        triage_scores[task] = t_score
        audit_scores[task] = a_score

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_triage = sum(triage_scores.values()) / len(triage_scores)
    avg_audit = sum(audit_scores.values()) / len(audit_scores)

    print(f"\n{'='*56}")
    print(f"  📊 PER-TASK SCORES:")
    for task in tasks:
        t_bar = "█" * int(triage_scores[task] * 10)
        a_bar = "█" * int(audit_scores[task] * 10)
        print(f"     {task:8s}: triage={triage_scores[task]:.4f} {t_bar}  audit={audit_scores[task]:.4f} {a_bar}")
    print(f"  {'─'*52}")
    print(f"  🔍 AVG TRIAGE SCORE: {avg_triage:.4f}")
    print(f"  🏁 AVG AUDIT SCORE:  {avg_audit:.4f}")
    print(f"{'='*56}\n")

    # Exit 0 = success (required by baseline-reproduces gate)
    sys.exit(0)


if __name__ == "__main__":
    main()