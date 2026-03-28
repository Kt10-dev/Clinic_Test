---
title: Clinical Trial Matchmaker
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Clinical Trial Matchmaker (OpenEnv)

![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-green)
![Domain](https://img.shields.io/badge/Domain-Healthcare-blue)
![Grader](https://img.shields.io/badge/Grader-Deterministic-orange)

## Overview & Motivation

**Problem:** Real-world matching of patients to clinical trials is incredibly slow, intensely manual, and prone to errors. Care teams must cross-reference complex patient health histories (EHRs) against highly dense trial documentation containing rigorous inclusion/exclusion criterion.

**Solution:** The **Clinical Trial Matchmaker** is a fully OpenEnv-compliant healthcare environment designed to evaluate an AI agent's ability to automate this process. An AI agent is placed securely inside a constrained, step-based workflow where it must carefully read unstructured medical histories, search medical catalogs, interpret exclusions, and make deterministic, safe clinical routing decisions.

This is a real-world task simulation (not a game) where failure represents potential clinical risk (false positives).

---

## Observation Space

At every `step()`, the environment returns a strongly-typed structured observation mapping:

- `patient_record`: The current patient's clinical EHR text.
- `search_results`: A JSON list of returned trials based on the agent's prior search queries.
- `current_trial_criteria`: The detailed textual inclusion/exclusion rules for a selected trial.
- `assigned_trial_id`: The ID of the currently assigned trial (if tentatively selected).
- `system_feedback`: Vital UI/Environment logs providing progressive feedback.

## Action Space

The Agent must navigate the task securely using a fully typed Pydantic schema:

- `search_trials(search_query)`: Queries the local database for disease-related trials.
- `read_criteria(trial_id)`: Selects a specific trial to view its extremely detailed rulebook.
- `assign_trial(trial_id)`: Attempts to route the current patient to a chosen trial.
- `mark_ineligible()`: Marks the patient as ineligible to participate, ensuring safety priority.
- `submit()`: Finalizes the Agent conclusion and triggers the episode boundary finish flag (`done=True`).

## Tasks and Graders

There are 3 core tasks evaluated deterministically by the `grader.py` module (Scores `0.0` to `1.0`).

| Task Level | Objective | Expected Difficulty |
| :--- | :--- | :--- |
| **Easy** | Match a patient to a trial using blunt inclusion data (e.g. basic disease match). | Very straightforward condition-to-trial parsing. |
| **Medium** | Handle hidden medical risks and strict rule contradictions. | More difficult: A patient may look like a perfect fit, but a past medical incidence breaches the exclusion criteria. |
| **Hard** | Temporal/Historical interpretation over years. | Extreme difficulty: Demands complex timeline calculus (e.g., "Must not have asthma for 5 years" while having a minor historical childhood occurrence). Model must grasp temporal negative statements. |

---

## Technical Specifications (Hackathon Requirements)

This project strictly adheres to the OpenEnv framework:
- **`openenv.yaml`** provided.
- **Typed Spaces**: Uses robust Pydantic Validation bounds (`models.py`).
- **Standardized Endpoints**: Perfectly conforms to `step()`, `reset()`, and `state()`.
- **Reproducible Baseline**: Provided via automated evaluation scripts.

### Baseline Scores (Reproducibility)
Using the standardized agent configurations, the environment accurately distinguishes baseline LLM capability logic yielding current maximum scores of:
- **Easy**: 1.0
- **Medium**: 1.0
- **Hard**: 1.0

---

## Setup & Usage (Core Testing)

### 1. Requirements
Ensure you have Python 3.10+ installed.

```bash
git clone https://github.com/yourusername/clinical-trial-env.git
cd clinical-trial-env
python -m venv .venv

# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Hackathon Automated Inference Testing
As required by the guidelines, you must use the root-level `inference.py` script to run the deterministic tests utilizing standard OpenEnv Huggingface variables.

Set your environment logic:
```bash
export API_BASE_URL="https://api.openai.com/v1"  # Or your specific Base API
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_hf_or_openai_key"
```

Execute the benchmark verification run:
```bash
python inference.py
```

### 3. Containerized Deploying
```bash
docker build -t clinical-trial-matchmaker .
docker run -p 8000:8000 clinical-trial-matchmaker
```

---

## ✨ Bonus Implementation: Bulk Batch-Processing API

In addition to traditional benchmark compatibility, the server (`main.py`) acts as a full-production Medical SaaS backend!
Deploy the interactive FastAPI interface using:
```bash
uvicorn main:app --reload
```
Navigate to `http://127.0.0.1:8000/docs` to test two real-world bonus additions:

1. **File Type Independence**: Upload `.pdf` and `.docx` reports to test the agent autonomously without manual copy/pasting.
2. **Bulk Uploading**: Utilizing the `POST /analyze-batch` endpoint, clinical coordinators can upload 50+ mixed files simultaneously to run parallel deterministic evaluations and map hospital volumes autonomously.
