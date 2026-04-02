---
title: Reproducibility Auditor
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
license: apache-2.0
short_description: "OpenEnv RL environment: audit ML experiments for reproducibility violations"
tags:
  - openenv
  - reinforcement-learning
  - ml-reproducibility
  - audit
  - benchmarking
---

# Reproducibility Auditor — OpenEnv Benchmark

The **Reproducibility Auditor** is a full-featured OpenEnv environment designed to evaluate AI agents' capability in identifying and fixing machine learning reproducibility vulnerabilities.

### Motivation
Modern ML research and production environments suffer from "reproducibility debt." Small omissions (like missing random seeds, unpinned dependencies, or non-deterministic CUDA operations) can lead to irreproducible results and wasted compute. This environment trains or evaluates agents to act as **Principal Systems Architects**, forensic auditors who ensure experiments are repeatable and mathematically deterministic.

**Real-world application:** An agent trained on this environment could power CI/CD checks that automatically flag reproducibility issues in ML codebases before results are published — similar to a linter, but for scientific correctness.

---

### Multi-Step Episode Design

Each episode consists of **2 steps**, providing trajectory-level reward signal:

```
reset() → Observation (code files)
   ↓
Step 1: TRIAGE → Agent identifies suspicious files + violation categories
   ↓         → Gets intermediate reward (F1 on file/category identification)
   ↓         → Gets actionable feedback (confirmed/rejected claims)
   ↓
Step 2: AUDIT → Agent submits full violation report (informed by triage feedback)
   ↓         → Gets final reward (per-violation detection score)
   ↓
Episode Done (terminated=True)
```

This 2-step design forces agents to develop genuine reasoning trajectories rather than one-shot pattern matching.

---

### Environment Specification

#### Observation space
*   **Type:** `text`
*   **Content:** A raw, concatenated dump of the experiment's source code files (e.g., `train.py`, `dataset.py`, `requirements.txt`).
*   **Step 2 enhancement:** After triage, the observation is augmented with feedback confirming/rejecting the agent's preliminary claims.

#### Action space (2-step)

**Step 1 — Triage Action:**
```json
{
  "suspicious_files": ["train.py", "requirements.txt"],
  "suspected_categories": ["random_seeds", "dependency_pinning"],
  "reasoning": "train.py lacks seed calls; requirements.txt has unpinned versions"
}
```

**Step 2 — Audit Action:**
```json
{
  "violations": [
    {
      "violation_type": "random.seed missing",
      "file_name": "train.py",
      "line_number": 5,
      "suggested_fix_code": "random.seed(42)"
    }
  ],
  "reproducibility_score": 0.3,
  "explanation": "Multiple seed calls are missing."
}
```

#### Violation Categories
`random_seeds` · `dependency_pinning` · `determinism_flags` · `environment_config` · `dataloader_reproducibility` · `model_initialization` · `configuration_management` · `multiprocessing` · `rng_initialization`

#### Reward Function
*   **Reward range:** `[0.0, 1.0]` (both triage and audit steps)
*   **Triage reward:** F1 score on file identification (60%) + category identification (40%)
*   **Audit reward:** Partial credit per violation. Each hit earns `1/N` where N = active violations.
*   **False-positive penalty:** Each false positive costs `1.0/N` — making keyword spamming strictly unprofitable.
*   **Per-field validation:** Keywords checked in `violation_type` and `suggested_fix_code` only. `file_name` validated against expected file per violation type.
*   **Bidirectional negation guard:** Dismissive phrases before/after keywords are rejected.

#### Dynamic Violation Injection
Each `reset()` randomly selects a subset of violations from the pool, generating unique experiment files with only those bugs present. This prevents memorisation and forces genuine code reasoning.

---

### Task Descriptions

| Task ID | Level | Violations Pool | Per-Episode | Description |
| :--- | :--- | :--- | :--- | :--- |
| `easy` | 🟢 Easy | 11 | 5–9 | Seeds, unpinned deps, PYTHONHASHSEED, cuDNN flags. |
| `medium` | 🟡 Medium | 8 | 4–6 | PyTorch determinism flags, DataLoader, Generator seeds, default_rng, Dropout. |
| `hard` | 🔴 Hard | 7 | 5–7 | Cross-file audits, version conflicts, multiprocessing, CUBLAS ordering. |

**Total violation pool: 26** — producing thousands of unique episode configurations.

---

### API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | GET | Health check — returns `{"status": "ok"}` with HTTP 200 |
| `/spec` | GET | Static environment metadata |
| `/reset` | POST | Reset environment, returns observation + active violations |
| `/step` | POST | Step 1: triage → feedback. Step 2: audit → final score |
| `/state` | GET | Current environment state with triage tracking |
| `/leaderboard` | GET | Scoring history with per-task statistics |

---

### Setup and Usage

#### 1. Required Environment Variables

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1/"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
```

#### 2. Local Development

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

#### 3. Using Docker

```bash
docker build -t reproducibility-auditor .
docker run -p 7860:7860 \
  -e HF_TOKEN=$HF_TOKEN \
  -e API_BASE_URL=$API_BASE_URL \
  -e MODEL_NAME=$MODEL_NAME \
  reproducibility-auditor
```

#### 4. Run Baseline Inference Script

```bash
# inference.py must be in root directory — uses OpenAI client for all LLM calls
# Runs 2-step episodes: triage → feedback → audit for each task
python inference.py
python inference.py --server http://localhost:7860
```

#### Baseline Scores

Tested with `Qwen/Qwen2.5-72B-Instruct` at `temperature=0.1`. 2-step episodes (triage → audit):

| Task | Triage (file/category F1) | Audit (violation detection) | Combined |
| :--- | :---: | :---: | :---: |
| `easy` | 0.85 | 0.89 | 0.87 |
| `medium` | 0.80 | 0.75 | 0.78 |
| `hard` | 0.70 | 0.63 | 0.67 |
| **Average** | **0.78** | **0.76** | **0.77** |

**Key observations:**
- Easy violations (seeds, dependencies) are reliably detected (>0.85 audit)
- Medium violations (DataLoader, determinism flags) show 0.75–1.00 variance across runs
- Hard violations (cross-file, CUBLAS, version conflicts) are the bottleneck — `worker_seed_cross_file` and `package_version_conflict` are the primary failure points
- Triage quality directly impacts audit scores — better file identification → better audit focus

#### 5. Pre-Submission Validation

```bash
# Start server first, then in a second terminal:
python validate.py
```

---

### Project Structure

```
├── env/
│   ├── __init__.py
│   ├── base_env.py          # Gymnasium environment (2-step episodes)
│   ├── openenv_wrapper.py   # OpenEnv spec-compliant wrapper
│   ├── generators.py        # Dynamic task file generators (30 violations)
│   └── graders/
│       ├── __init__.py
│       ├── easy_grader.py   # 11-violation grader with file validation
│       ├── medium_grader.py # 8-violation grader with file validation
│       └── hard_grader.py   # 7-violation grader with file validation
├── server.py                # FastAPI OpenEnv server
├── app.py                   # Hugging Face Spaces / Docker entrypoint
├── inference.py             # Baseline inference script (root directory, required)
├── openenv.yaml             # OpenEnv spec file
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── requirements.txt
└── validate.py              # Pre-submission validation suite
```
