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

---

### Environment Specification

#### Observation space
*   **Type:** `text`
*   **Content:** A raw, concatenated dump of the experiment's source code files (e.g., `train.py`, `dataset.py`, `requirements.txt`).

#### Action space
*   **Type:** `json` (Validated via Pydantic model `AuditAction`)
*   **Schema:**
    *   `violations`: `list[ViolationObject]` — Each with `violation_type`, `file_name`, `line_number`, `suggested_fix_code`.
    *   `reproducibility_score`: `float` (0.0–1.0) — Agent's self-assessed score.
    *   `explanation`: `str` — Human-readable summary of the audit.

#### Reward Function
The environment uses a **deterministic keyword-matching grader** with:
*   **Reward range:** `[0.0, 1.0]`
*   **Partial credit:** Each correctly identified violation earns `1/N` where N is the number of active violations in that episode.
*   **False-positive penalty:** Claiming violations that are not present in the episode reduces the score by `0.5/N` per false positive.
*   **Per-field validation:** Keywords are checked in `violation_type` and `suggested_fix_code` only — not in `explanation` or `file_name`.
*   **Bidirectional negation guard:** Dismissive phrases before or after the keyword are rejected.

#### Dynamic Violation Injection
Each call to `reset()` randomly selects a subset of violations from the pool, generating unique experiment files with only those bugs present. This prevents memorisation and forces genuine code reasoning.

---

### Task Descriptions

| Task ID | Level | Violations Pool | Per-Episode | Description |
| :--- | :--- | :--- | :--- | :--- |
| `easy` | 🟢 Easy | 12 | 5–9 | Seeds, unpinned deps, PYTHONHASHSEED, cuDNN flags, thread count. |
| `medium` | 🟡 Medium | 8 | 4–6 | PyTorch determinism flags, DataLoader, Generator seeds, default_rng, Dropout. |
| `hard` | 🔴 Hard | 10 | 5–8 | Cross-file audits, version conflicts, multiprocessing, late hashseed, CUBLAS ordering. |

**Total violation pool: 30** — producing thousands of unique episode configurations.

---

### API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | GET | Health check — returns `{"status": "ok"}` with HTTP 200 |
| `/spec` | GET | Static environment metadata |
| `/reset` | POST | Reset environment, returns observation + active violations |
| `/step` | POST | Submit action, returns reward + breakdown |
| `/state` | GET | Current environment state |
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
python inference.py
python inference.py --server http://localhost:7860
```

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
│   ├── base_env.py          # Gymnasium environment
│   ├── openenv_wrapper.py   # OpenEnv spec-compliant wrapper
│   ├── generators.py        # Dynamic task file generators (30 violations)
│   └── graders/
│       ├── __init__.py
│       ├── easy_grader.py   # 12-violation grader
│       ├── medium_grader.py # 8-violation grader
│       └── hard_grader.py   # 10-violation grader
├── tasks/                   # Reference static task files
├── server.py                # FastAPI OpenEnv server
├── app.py                   # Hugging Face Spaces / Docker entrypoint
├── inference.py             # Baseline inference script (root directory, required)
├── openenv.yaml             # OpenEnv spec file
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── validate.py              # Pre-submission validation suite
```
