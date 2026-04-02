# Reproducibility Auditor — Deep Project Analysis

**Date:** 2026-04-01
**Total Files:** 15 (excluding venv, .git, __pycache__)
**Total Lines of Code:** ~2,800
**Violation Pool:** 26 (11 easy + 8 medium + 7 hard)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [File-by-File Analysis](#2-file-by-file-analysis)
3. [Data Flow](#3-data-flow)
4. [Known Flaws and Issues](#4-known-flaws-and-issues)
5. [Improvements](#5-improvements)
6. [Variance Analysis](#6-variance-analysis)

---

## 1. Architecture Overview

```
                    ┌─────────────────┐
                    │   inference.py  │ ← LLM agent (OpenAI client)
                    └────────┬────────┘
                             │ HTTP (POST /reset, /step)
                             ▼
                    ┌─────────────────┐
                    │   server.py     │ ← FastAPI REST server
                    │   (app.py)      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ openenv_wrapper │ ← Pydantic schema validation
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   base_env.py   │ ← Gymnasium env (triage → audit)
                    ├─────────────────┤
                    │  generators.py  │ ← Dynamic file generation
                    │  graders/*.py   │ ← Keyword-based scoring
                    └─────────────────┘
```

**The system has 3 layers:**
1. **HTTP layer** (`server.py`, `app.py`) — receives requests, routes to wrapper
2. **Spec layer** (`openenv_wrapper.py`) — validates actions via Pydantic, tracks episode state
3. **Environment layer** (`base_env.py`, `generators.py`, `graders/`) — generates tasks, scores actions

---

## 2. File-by-File Analysis

### 2.1 `env/base_env.py` (522 lines) — Core Gymnasium Environment

**Purpose:** The main RL environment. Implements `reset()`, `step()`, and `render()` following gymnasium conventions.

**Key structures:**
- `TASK_METADATA` — difficulty, focus areas, expected violation counts per task
- `VIOLATION_FILE_MAP` — maps each of 26 violation IDs to the file where it appears
- `VIOLATION_CATEGORY_MAP` — maps each violation ID to one of 9 categories
- `ALL_CATEGORIES` — sorted list of 9 unique category names
- `_DECOY_COMMENTS` — 10 decoy comments injected during presentation randomization
- `ReproducibilityAuditorEnv` — the gymnasium `Env` subclass

**Episode flow:**
1. `reset()` — selects random violations via `select_violations()`, generates files via `generate_files()`, applies presentation randomization, formats observation text
2. `step(action)` — auto-detects triage vs audit from JSON keys
3. `_triage_step()` — scores file/category F1, returns feedback appended to observation
4. `_audit_step()` — delegates to grader, returns final score

**Scoring:**
- Triage: 0.6 × file_F1 + 0.4 × category_F1
- Audit: delegator to task-specific grader

**Issues found:**
- `TASKS_DIR` (line 30) points to deleted `tasks/` directory — dead code
- `AuditReport` dataclass (line 148) is defined but never used
- `re` import (line 19) is unused
- `np` import (line 21) is unused in this file
- `_DECOY_COMMENTS` could contain keywords that trigger grader false positives (e.g., "Verified on CUDA 12.1 + cuDNN 8.9" contains "cudnn")
- `expected_violations` in metadata doesn't match actual pool sizes (easy says 12, actual is 11)

### 2.2 `env/openenv_wrapper.py` (288 lines) — OpenEnv Spec Wrapper

**Purpose:** Wraps `ReproducibilityAuditorEnv` in the exact interface OpenEnv expects. Defines Pydantic schemas for actions and results.

**Key classes:**
- `TriageAction` — schema for step 1 (suspicious_files, suspected_categories, reasoning)
- `ViolationObject` — schema for a single violation claim
- `AuditAction` — schema for step 2 (violations list, reproducibility_score, explanation)
- `StepResult` — returned by every `step()` call
- `ResetResult` — returned by `reset()`
- `EnvState` — full serializable state for `/state` endpoint
- `ReproducibilityEnvOpenEnv` — the main wrapper class

**Action type detection:**
- `_detect_triage()` — checks for `suspicious_files` or `suspected_categories` keys
- `_coerce_triage()` / `_coerce_action()` — accepts string, dict, or Pydantic model, returns validated JSON string

**Issues found:**
- `_detect_triage()` and `_coerce_action()` have type narrowing errors (LSP warnings) — `AuditAction` could be passed to `_coerce_triage()` because the `is_triage` boolean is computed before coercion
- `spec()` returns `observation_type: "structured_text"` but `openenv.yaml` says `"text"` — inconsistency
- `step()` has duplicated state tracking with `base_env.py` — step count tracked in both places

### 2.3 `env/generators.py` (353 lines) — Dynamic Task File Generators

**Purpose:** Generates synthetic ML experiment files with specific violations injected.

**Violation pools:**
- `EASY_ALL_VIOLATIONS` — 11 violations (seeds, deps, cuDNN)
- `MEDIUM_ALL_VIOLATIONS` — 8 violations (determinism, DataLoader, Generator)
- `HARD_ALL_VIOLATIONS` — 7 violations (cross-file, CUBLAS, version conflicts)

**Key functions:**
- `select_violations(task, rng)` — random subset of violations for the episode
- `generate_files(task, active, rng)` — generates file content with line numbers
- `_add_line_numbers(content)` — prepends `N: ` to each line
- `_gen_easy()`, `_gen_medium()`, `_gen_hard()` — task-specific generators

**Generation logic:**
- Each generator checks `if "violation_id" in active` to decide whether to include or omit specific code patterns
- The inversion: `not in active` means the fix IS present, `in active` means the bug IS present

**Issues found:**
- Docstring says "30 total violations" but actual count is 26
- `_gen_hard()` uses `torch.manual_seed(seed)` in `set_seeds()` unconditionally — this means `missing_torch_seed` violation is never injected in hard tasks (torch seed is always present)
- No `rng` parameter used in `_gen_medium()` — the `rng` arg is accepted but ignored
- `_gen_hard()` has `config_yaml_override` logic that's subtle: when active, `np.random.seed(seed)` and `random.seed(seed)` are omitted from `set_seeds()`, but the function name `set_seeds` implies all seeds should be set
- The `_EASY_BODY` has hardcoded `print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")` which could confuse an agent about what the "real" training loop looks like

### 2.4 `env/graders/easy_grader.py` (192 lines) — Easy Task Grader

**Purpose:** Scores agent's violation claims for the easy task (11 violations).

**Scoring mechanism:**
1. Parse JSON action into `claim_entries` — list of `(texts, file_name)` tuples
2. For each violation ID, check if any claim's `violation_type` or `suggested_fix_code` contains a matching keyword
3. Validate `file_name` against `_EASY_FILE_MAP`
4. Apply bidirectional negation guard (25 chars before/after)
5. Score: `(hits - 1.0 × false_positives) / num_active`, clamped to [0, 1]

**Keyword sets:**
- Seed violations: 5-8 keywords each
- Dependency violations: 4 keywords each (generated via loop)
- Determinism violations: 5-7 keywords each

**Negation guard:**
- Before: rejects `already`, `no need`, `skip`, `skipping`, `bypass`, `ignore`, `don't need`, `shouldn't`
- After: rejects `not required`, `not needed`, `not necessary`, `unnecessary`, `irrelevant`, `not an issue`, `already set/present/handled/configured`
- Deliberately does NOT reject bare `not` or `without` because LLMs use them in legitimate violation reports

**Issues found:**
- `missing_random_seed` keywords include `"random.seed"` which is a substring of `"np.random.seed"` — an agent reporting `"np.random.seed missing"` would also trigger `missing_random_seed` detection (false match within same claim)
- `line_number` field from the agent is completely ignored — only `violation_type`, `suggested_fix_code`, and `file_name` matter
- The negation guard uses regex with word boundaries (`\b`) which may not work correctly in all edge cases with punctuation

### 2.5 `env/graders/medium_grader.py` (154 lines) — Medium Task Grader

**Purpose:** Scores agent's claims for the medium task (8 violations).

**Structure:** Nearly identical to easy grader but with different keyword sets and file map (all violations in `train.py`).

**Keyword sets:**
- `dataloader_shuffle_no_seed`: `shuffle`, `dataloader`, `shuffle=true`, `shuffle = true`, `shuffle=True`
- `missing_deterministic_flag`: `use_deterministic_algorithms`, `deterministic algorithms`, `deterministic_algorithms`, `torch.use_deterministic`, `deterministic_algorithms missing`
- `missing_dropout_seed`: `dropout without seed`, `nn.dropout without seed`, `dropout seed guard`, `dropout determinism`, `dropout not seeded`, `dropout randomness`

**Issues found:**
- `_MEDIUM_FILE_MAP` maps ALL violations to `train.py` — this means file validation always passes for any claim with `file_name="train.py"`, making the file check meaningless for this task
- `dataloader_shuffle_no_seed` keywords `"shuffle"` and `"dataloader"` are very broad — could match unrelated mentions
- `_extract_claim_entries()` is a module-level function called from a class method — works at runtime but triggers LSP warnings
- `_is_valid_claim()` is duplicated across all 3 graders — code duplication

### 2.6 `env/graders/hard_grader.py` (178 lines) — Hard Task Grader

**Purpose:** Scores agent's claims for the hard task (7 violations, cross-file).

**Structure:** Similar to other graders but with two-pass detection for `worker_seed_cross_file`:
1. First pass: requires `has_dataset` context (file name or text mentions "dataset")
2. Second pass: very specific keywords that don't need dataset context

**Special handling:**
- `worker_seed_cross_file` — broadened keywords including `seed_worker`, `worker seeding`, `worker id seed`, `worker process seed`
- `package_version_conflict` — 14 keywords including version numbers, "conflicting versions", "torch and torchvision"
- `config_yaml_override` — 21 keywords (most extensive set in the project)

**Issues found:**
- `model_weight_init_seed` keywords include `"model.py"` — this means ANY violation claim in `model.py` that mentions "model.py" in its `violation_type` will trigger this detection, even if it's about a different issue
- `multiprocessing_no_seed` keywords `"multiprocessing"` and `"pool("` could match in non-reproducibility contexts
- `_is_valid_claim()` is duplicated from other graders (3rd copy)

### 2.7 `server.py` (175 lines) — FastAPI HTTP Server

**Purpose:** Exposes the environment over HTTP following the OpenEnv REST spec.

**Endpoints:**
- `GET /` — root ping (HTTP 200 for HF Spaces)
- `GET /health` — health check with task list
- `GET /spec` — static environment metadata
- `POST /reset` — reset environment, return observation
- `POST /step` — submit action, return reward + info
- `GET /state` — current environment state
- `GET /leaderboard` — scoring history with per-task statistics

**Design:**
- One `ReproducibilityEnvOpenEnv` instance per task (created at startup via lifespan)
- Leaderboard stored in-memory (list of dicts) — lost on restart
- `StepRequest.action` is `dict[str, Any]` — passes raw dict to wrapper which handles validation

**Issues found:**
- Leaderboard is in-memory only — lost on server restart
- No rate limiting or authentication on endpoints
- `StepRequest.action` type is `dict[str, Any]` instead of `Union[TriageAction, AuditAction]` — means Pydantic validation at the HTTP layer is minimal; actual validation happens inside the wrapper
- No CORS configuration — would fail if a frontend tried to connect
- `/state` endpoint returns task-specific state but doesn't validate the task parameter against known tasks (only checks against `envs` dict)

### 2.8 `app.py` (12 lines) — Hugging Face Spaces Entrypoint

**Purpose:** Re-exports the FastAPI `app` from `server.py` so `uvicorn app:app` works.

**Issues found:**
- The `__main__` block runs `uvicorn.run("server:app", ...)` but the CMD in Dockerfile is `uvicorn app:app` — these are equivalent because `app.py` re-exports `server.app`, but the inconsistency is confusing
- No error handling if `server` module fails to import

### 2.9 `inference.py` (390 lines) — Baseline Inference Script

**Purpose:** The mandatory baseline script that runs an LLM against the environment to produce reproducible scores.

**Flow:**
1. Validate env vars (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`)
2. Ping `/health` endpoint
3. Build OpenAI client
4. For each task (easy, medium, hard):
   a. `POST /reset` — get observation
   b. LLM triage call — get suspicious files/categories
   c. `POST /step` — submit triage, get feedback
   d. LLM audit call — get violation report (with triage feedback in prompt)
   e. `_sanitize_audit_dict()` — fix LLM typos in keys
   f. `POST /step` — submit audit, get final score
5. Print summary

**Key components:**
- `TRIAGE_PROMPT` — system prompt for step 1 (file/category identification)
- `AUDIT_PROMPT` — system prompt for step 2 (detailed violation report with checklist)
- `_sanitize_audit_dict()` — maps 13 common typo keys to correct names
- `_parse_llm_json()` — strips markdown fences from LLM response

**Issues found:**
- `AUDIT_PROMPT` still mentions `"deterministic_algorithms without CUBLAS"` and `"manual_seed instead of manual_seed_all"` (lines 146-147) — these violations were removed from the pool. The LLM may try to report them, wasting tokens and potentially confusing the model.
- `AUDIT_PROMPT` mentions `"Dropout without seed guard"` (line 137) which has very tight keywords in the medium grader — the prompt suggests `"nn.Dropout without seed"` but the grader's keywords were changed to `"dropout without seed"`, `"nn.dropout without seed"`, etc.
- Temperature is 0.1 for both calls — very low, good for reproducibility but may reduce the LLM's ability to catch subtle violations
- No retry logic for LLM API failures
- `_sanitize_audit_dict()` only handles key typos, not value typos (e.g., `file_nmae: "tran.py"` would be fixed to `file_name: "tran.py"` — the key is fixed but the value is still wrong)

### 2.10 `validate.py` (290 lines) — Pre-Submission Validator

**Purpose:** Automated validation script that checks all pre-submission checklist items.

**6 check groups:**
1. Deployment & Health — HTTP 200, all 3 tasks loaded
2. OpenEnv Spec Compliance — `/spec` returns correct structure
3. Multi-Step reset/step — functional test for each task
4. Grader Score Variance — empty vs correct action produces different scores
5. State Endpoint — `/state` returns expected fields
6. Trajectory Signal — good vs bad triage produces different rewards

**Issues found:**
- `VIOLATION_KEYWORDS` dict duplicates the keyword maps from graders — if grader keywords change, this must be updated manually
- Test uses `from env.base_env import VIOLATION_FILE_MAP, VIOLATION_CATEGORY_MAP` inside the function — this import should be at module level
- Step 1 test uses empty triage action — tests that the system doesn't crash but doesn't test that a good triage gets more reward than a bad one (test 6 does this separately)
- No test for the `_sanitize_audit_dict` function

### 2.11 `test_graders.py` (295 lines) — Unit Tests

**Purpose:** Tests grader behavior, dynamic violation selection, file generation, and the OpenEnv wrapper.

**7 test functions:**
1. `test_medium_grader()` — gaming vs honest detection
2. `test_hard_grader()` — honest detection
3. `test_false_positive_penalty()` — shotgun vs precise agent
4. `test_dynamic_violations()` — different seeds → different subsets
5. `test_generated_files()` — files reflect active violations
6. `test_openenv_wrapper_dynamic()` — end-to-end with correct file names
7. `test_file_name_validation()` — wrong file → 0 reward

**Issues found:**
- `test_generated_files()` uses `strip_lineno()` helper that's defined inside the function — works but unusual
- No test for medium grader's file validation (all files are `train.py` so it's trivially passing)
- No test for the `_sanitize_audit_dict` function
- No test for the triage scoring function
- `test_hard_grader()` only tests `worker_seed_cross_file` — doesn't test other hard violations

### 2.12 `openenv.yaml` (185 lines) — OpenEnv Spec File

**Purpose:** Static metadata file describing the environment.

**Issues found:**
- Line 5: "30 violation types" — should be 26
- Line 178: `# 9-violation grader` in README project structure — should be 7
- `observation_type` is `"text"` here but `"structured_text"` in `spec()` — inconsistency

### 2.13 `Dockerfile` (36 lines) — Container Definition

**Purpose:** Builds a containerized version of the environment.

**Issues found:**
- Uses `python:3.11-slim` — good, lightweight
- Installs `gcc` — needed for some Python C extensions but increases image size
- No `.gitignore` copy — not needed in container
- No `validate.py` or `test_graders.py` copied — not needed in production
- HEALTHCHECK uses `curl` which was installed — correct

### 2.14 `docker-compose.yml` (21 lines) — Docker Compose Config

**Purpose:** Simplifies local Docker development.

**Issues found:**
- Uses `version: "3.9"` which is deprecated in newer Docker Compose versions
- Environment variables use shell expansion (`${VAR:-default}`) — correct

### 2.15 `.dockerignore` (9 lines) — Docker Build Exclusions

**Purpose:** Excludes unnecessary files from Docker builds.

**Issues found:**
- Excludes `tasks/` and `agent/` which were deleted — fine, prevents any re-creation issues
- Should also exclude `test_graders.py`, `validate.py`, `.gitignore`, `.kilo/`

### 2.16 `requirements.txt` (8 lines) — Python Dependencies

**Dependencies:**
- `fastapi==0.111.0` — web framework
- `uvicorn[standard]==0.29.0` — ASGI server
- `pydantic==2.7.1` — data validation
- `gymnasium==0.29.1` — RL environment interface
- `requests==2.31.0` — HTTP client (used by inference.py)
- `pyyaml==6.0.1` — YAML parsing (unused — generators create YAML as string)
- `openai==1.30.1` — OpenAI client
- `python-dotenv==1.0.1` — .env file loading

**Issues found:**
- `pyyaml` is listed but never imported by any file in the project — dead dependency
- `gymnasium` is used only by `base_env.py` for the `gym.Env` base class and spaces — the gymnasium API is used minimally (only `Text` spaces)
- No pinned sub-dependencies — could break with transitive updates

### 2.17 `README.md` (193 lines) — Project Documentation

**Issues found:**
- Line 178: says `hard_grader.py # 9-violation grader` — should be 7
- Line 108: says "Total violation pool: 26" — correct
- HF Spaces YAML frontmatter at top — correct format

---

## 3. Data Flow

### Reset Flow
```
POST /reset {"task": "easy"}
  → server.py: envs["easy"].reset()
  → openenv_wrapper.py: _inner.reset()
  → base_env.py: 
      1. select_violations("easy", rng) → set of 5-9 violation IDs
      2. generate_files("easy", violations, rng) → {"train.py": "...", "requirements.txt": "..."}
      3. _randomize_presentation(files, rng) → shuffled order + decoy comments
      4. _format_observation(files, "easy") → structured text with metadata header
  → returns ResetResult(observation=text, info={active_violations: [...], ...})
```

### Step 1 (Triage) Flow
```
POST /step {"task": "easy", "action": {"suspicious_files": [...], "suspected_categories": [...], ...}}
  → server.py: envs["easy"].step(action)
  → openenv_wrapper.py: _detect_triage(action) → True
  → openenv_wrapper.py: _coerce_triage(action) → validated JSON string
  → base_env.py: _triage_step(json_str)
      1. Parse JSON
      2. _score_triage(triage) → F1 on files (60%) + categories (40%)
      3. Build feedback text
      4. Append feedback to observation
  → returns StepResult(observation=enhanced_text, reward=triage_score, terminated=False, ...)
```

### Step 2 (Audit) Flow
```
POST /step {"task": "easy", "action": {"violations": [...], "reproducibility_score": 0.5, ...}}
  → server.py: envs["easy"].step(action)
  → openenv_wrapper.py: _detect_triage(action) → False
  → openenv_wrapper.py: _coerce_action(action) → validated JSON string
  → base_env.py: _audit_step(json_str)
      1. grader.score(json_str, active_violations)
      2. grader parses JSON, builds claim_entries
      3. For each violation_id: check keywords + file_name
      4. Calculate: (hits - FP) / num_active
  → returns StepResult(observation=text, reward=audit_score, terminated=True, ...)
```

---

## 4. Known Flaws and Issues

### 4.1 Critical Issues

| # | Issue | Impact | File |
|---|-------|--------|------|
| 1 | **`AUDIT_PROMPT` mentions removed violations** — `"deterministic_algorithms without CUBLAS"` and `"manual_seed instead of manual_seed_all"` are still in the prompt but no longer in the pool. Wastes LLM tokens and may confuse the model. | High | inference.py:146-147 |
| 2 | **`expected_violations` mismatch** — `TASK_METADATA["easy"]["expected_violations"]` is 12 but the actual pool has 11. Could confuse agents reading the metadata. | Medium | base_env.py:38 |
| 3 | **`openenv.yaml` says "30 violation types"** — actual count is 26. Inconsistent documentation. | Medium | openenv.yaml:5 |
| 4 | **`_DECOY_COMMENTS` can trigger grader** — comment `"Verified on CUDA 12.1 + cuDNN 8.9"` contains `"cudnn"` which matches grader keywords for cuDNN violations. | Low-Medium | base_env.py:140 |
| 5 | **Medium grader file validation is trivial** — ALL medium violations map to `train.py`, so any claim with `file_name="train.py"` passes file validation. | Low | medium_grader.py:25-34 |

### 4.2 Design Issues

| # | Issue | File |
|---|-------|------|
| 6 | **`_is_valid_claim()` duplicated 3 times** — identical logic in easy, medium, and hard graders. Should be a shared utility. | graders/*.py |
| 7 | **`line_number` completely ignored** — graders never check if the agent's reported line number matches the actual violation location. | All graders |
| 8 | **Keyword substring false positives** — `"random.seed"` is a substring of `"np.random.seed"`, so a claim mentioning `"np.random.seed missing"` also triggers `missing_random_seed`. | easy_grader.py:82 |
| 9 | **In-memory leaderboard** — lost on restart. No persistence. | server.py:29 |
| 10 | **No retry logic in inference.py** — if the LLM API call fails once, the entire run fails. | inference.py |
| 11 | **`pyyaml` dependency unused** — listed in requirements.txt but never imported. | requirements.txt |
| 12 | **`TASKS_DIR` points to deleted directory** — dead reference. | base_env.py:30 |
| 13 | **`AuditReport` dataclass unused** — defined but never instantiated. | base_env.py:148 |
| 14 | **Duplicated state tracking** — step count tracked in both `base_env.py` and `openenv_wrapper.py`. | base_env.py, openenv_wrapper.py |
| 15 | **`spec()` returns `"structured_text"` but yaml says `"text"`** — inconsistency in observation type. | openenv_wrapper.py:206, openenv.yaml:42 |

### 4.3 Variance Sources

| # | Source | Effect |
|---|--------|--------|
| 16 | **Dynamic violation selection** — different episodes have different violations active. The LLM must re-analyze each time. | Core feature, but increases variance |
| 17 | **Presentation randomization** — file order, decoy comments, whitespace changes per episode. | Core feature |
| 18 | **LLM non-determinism** — even at temperature=0.1, the LLM produces different outputs. | Unavoidable |
| 19 | **Keyword matching is fragile** — if the LLM describes a violation differently than expected, it's missed. | Grader design |
| 20 | **Removed violations reduce pool** — hard went from 10→7 violations, making each violation worth more (1/7 vs 1/10). | Side effect of removals |

---

## 5. Improvements

### 5.1 High Priority (Score Impact)

1. **Clean AUDIT_PROMPT** — Remove references to deleted violations (`deterministic_without_cublas`, `incomplete_cuda_seed`). Add explicit instruction to check `file_name` against the correct file.

2. **Fix `expected_violations`** — Update `TASK_METADATA["easy"]["expected_violations"]` from 12 to 11.

3. **Fix `openenv.yaml` docstring** — Change "30 violation types" to "26".

4. **Add line number validation** — For violations in files with line numbers, check if the agent's reported line is within ±5 of the actual violation line. This would significantly increase grader accuracy.

5. **Decouple decoy comments from grader keywords** — Remove any decoy comment that contains words like "cudnn", "seed", "deterministic" that could trigger false positives.

### 5.2 Medium Priority (Code Quality)

6. **Extract `_is_valid_claim()` to shared module** — Create `env/graders/utils.py` with the negation guard logic, import from all 3 graders.

7. **Add retry logic to inference.py** — Wrap LLM calls in a retry decorator (3 attempts, exponential backoff).

8. **Persist leaderboard** — Write to a JSON file or use SQLite instead of in-memory list.

9. **Remove dead code** — `TASKS_DIR`, `AuditReport`, `re` import, `np` import, `pyyaml` dependency.

10. **Fix `_DECOY_COMMENTS` to be keyword-safe** — Replace "cuDNN 8.9" with "cuDNN 8.x", remove any seed-related decoys.

### 5.3 Low Priority (Polish)

11. **Consolidate state tracking** — Have `openenv_wrapper.py` read state from `_inner` instead of duplicating tracking.

12. **Fix `spec()` observation type** — Change to `"text"` to match `openenv.yaml`.

13. **Add `.dockerignore` entries** — Exclude `test_graders.py`, `validate.py`, `.gitignore`.

14. **Update README project structure** — Fix "9-violation grader" to "7-violation grader".

15. **Add type hints to generator functions** — `_gen_easy()`, `_gen_medium()`, `_gen_hard()` lack return type hints.

---

## 6. Variance Analysis

### Observed Run Scores (from user's data)
```
Run 1:  easy=1.000  medium=1.000  hard=0.833  → avg 0.944
Run 2:  easy=1.000  medium=0.750  hard=0.400  → avg 0.717
Run 3:  easy=0.889  medium=1.000  hard=0.600  → avg 0.830
Run 4:  easy=0.667  medium=0.500  hard=0.750  → avg 0.639
```

### Variance Sources by Task

**Easy (range: 0.667-1.000, spread: 0.333):**
- Dynamic violations: 5-9 of 11 active per episode
- LLM must identify which of the 4 files have violations
- Seed violations are generally well-detected; dependency pinning is where LLMs struggle

**Medium (range: 0.500-1.000, spread: 0.500):**
- Dynamic violations: 4-6 of 8 active
- All violations in single `train.py` file — reduces file confusion
- `dataloader_shuffle_no_seed` and `missing_dropout_seed` have tighter keywords → higher failure rate

**Hard (range: 0.400-0.833, spread: 0.433):**
- Dynamic violations: 5-7 of 7 active — nearly all violations are active
- Multi-file (train.py, dataset.py, model.py, requirements.txt, config.yaml)
- `worker_seed_cross_file` was the persistent failure (now broadened)
- `package_version_conflict` was inconsistent (now has 14 keywords)

### Floor/Ceiling After Latest Fixes
- Removed `incomplete_cuda_seed` (3/4 failure rate) → hard floor should rise
- Broadened `package_version_conflict` keywords → hard consistency should improve
- Broadened `worker_seed_cross_file` keywords → hard detection rate should improve
- Expected new range: 0.75-0.95 average across all tasks
