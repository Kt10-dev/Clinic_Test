import time
import asyncio
import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger("clinical_trial_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import io
import json
import datetime

from environment import ClinicalTrialEnv, PATIENTS_DB
from grader import evaluate_episode
from models import Action

app = FastAPI(title="Clinical Trial Matchmaker Env")
@app.get("/")
def home():
    return {"message": "Clinical Trial Matchmaker API is Running!", "docs": "/docs"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/schema")
def get_schema():
    return Action.model_json_schema()

@dataclass
class SessionRuntime:
    env: ClinicalTrialEnv
    lock: Any = field(default_factory=Lock)
    last_accessed: float = field(default_factory=time.time)


env_sessions: Dict[str, SessionRuntime] = {}
sessions_lock = Lock()
SESSION_TIMEOUT = 3600  # 1 hour in seconds


def _cleanup_expired_sessions() -> None:
    now = time.time()
    expired = []
    with sessions_lock:
        for sid, runtime in env_sessions.items():
            if now - runtime.last_accessed > SESSION_TIMEOUT:
                expired.append(sid)
        for sid in expired:
            del env_sessions[sid]
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired sessions.")


def _get_session_env(session_id: Optional[str] = None) -> SessionRuntime:
    _cleanup_expired_sessions()
    
    with sessions_lock:
        if session_id is None:
            if len(env_sessions) == 1:
                runtime = next(iter(env_sessions.values()))
                runtime.last_accessed = time.time()
                return runtime
            elif len(env_sessions) == 0:
                raise HTTPException(status_code=400, detail="No active session. Please call /reset first.")
            else:
                raise HTTPException(status_code=400, detail="Multiple sessions exist. Must provide session_id.")
        
        runtime = env_sessions.get(session_id)

    if runtime is None:
        logger.warning(f"Session {session_id} not found or expired.")
        raise HTTPException(status_code=404, detail="Session not found.")

    runtime.last_accessed = time.time()
    return runtime


class ResetRequest(BaseModel):
    task_difficulty: str = "easy"

@app.post("/reset")
def reset_env(req: Optional[ResetRequest] = None) -> Dict[str, Any]:
    _cleanup_expired_sessions()
    
    difficulty = req.task_difficulty if req else "easy"
    
    if difficulty not in PATIENTS_DB:
        logger.warning(f"Invalid task difficulty requested: {difficulty}")
        raise HTTPException(
            status_code=400,
            detail="Invalid task difficulty. Choose easy, medium, or hard.",
        )

    env = ClinicalTrialEnv()
    observation = env.reset(difficulty)
    session_id = str(uuid4())

    with sessions_lock:
        env_sessions[session_id] = SessionRuntime(env=env)
        
    logger.info(f"Started new session {session_id} with difficulty {difficulty}")

    return {
        "session_id": session_id,
        "observation": observation.model_dump(),
    }


@app.post("/step")
def step_env(action: Action, session_id: Optional[str] = None) -> Dict[str, Any]:
    runtime = _get_session_env(session_id)
    with runtime.lock:
        observation, reward, done, info = runtime.env.step(action)

    return {
        "observation": observation.model_dump(),
        "reward": reward.value,
        "done": done,
        "info": info,
    }


@app.get("/state")
def get_state(session_id: Optional[str] = None) -> Dict[str, Any]:
    runtime = _get_session_env(session_id)
    with runtime.lock:
        return runtime.env.state()


@app.get("/tasks")
def get_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "id": "easy",
                "description": "Match a patient with straightforward criteria.",
            },
            {
                "id": "medium",
                "description": "Match a patient considering complex exclusion criteria.",
            },
            {
                "id": "hard",
                "description": "Match a patient with complex timeline and prior treatment logic.",
            },
        ],
        "action_schema": Action.model_json_schema(),
    }


@app.get("/grader")
def get_grader_score(session_id: Optional[str] = None) -> Dict[str, float]:
    runtime = _get_session_env(session_id)
    with runtime.lock:
        score = evaluate_episode(runtime.env.state(), runtime.env.expected_trial)
    return {"score": score}


@app.delete("/session")
def close_session(session_id: Optional[str] = None) -> Dict[str, str]:
    with sessions_lock:
        if session_id is None:
            if len(env_sessions) == 1:
                session_id = next(iter(env_sessions.keys()))
            else:
                raise HTTPException(status_code=400, detail="Multiple sessions exist. Must provide session_id.")
        if session_id not in env_sessions:
            logger.warning(f"Attempted to close non-existent session {session_id}")
            raise HTTPException(status_code=404, detail="Session not found.")
        del env_sessions[session_id]
        
    logger.info(f"Closed session {session_id}")

    return {"status": "deleted", "session_id": session_id}


async def _extract_text_from_upload(file: UploadFile) -> str:
    file_bytes = await file.read()
    filename = (file.filename or "unknown").lower()
    
    extracted_text = ""
    try:
        if filename.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                extracted_text += page.extract_text() + "\n"
        elif filename.endswith(".docx"):
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            extracted_text = "\n".join([para.text for para in doc.paragraphs])
        else: # Fallback to txt
            extracted_text = file_bytes.decode("utf-8")
    except Exception as e:
        logger.error(f"Error parsing file {filename}: {e}")
        raise ValueError(f"Failed to parse file: {str(e)}")
        
    return extracted_text

@app.post("/analyze-report")
async def analyze_patient_report(file: UploadFile = File(...)) -> Dict[str, Any]:
    _cleanup_expired_sessions()
    
    try:
        extracted_text = await _extract_text_from_upload(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file contained no extractable text.")
        
    logger.info(f"Successfully extracted {len(extracted_text)} characters from {file.filename}")
    
    try:
        from baseline import run_agent_on_custom_record
        temp_env = ClinicalTrialEnv()
        result = run_agent_on_custom_record(env=temp_env, record_text=extracted_text)
        
        return {
            "status": "success",
            "filename": file.filename,
            "analysis_result": result
        }
    except Exception as exc:
        logger.error(f"Error analyzing custom report: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/analyze-batch")
async def analyze_batch_reports(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    _cleanup_expired_sessions()
    
    from baseline import run_agent_on_custom_record
    
    batch_results = []
    
    for idx, file in enumerate(files):
        try:
            extracted_text = await _extract_text_from_upload(file)
            
            if not extracted_text.strip():
                batch_results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": "No extractable text found."
                })
                continue
                
            temp_env = ClinicalTrialEnv()
            result = run_agent_on_custom_record(env=temp_env, record_text=extracted_text)
            
            batch_results.append({
                "filename": file.filename,
                "status": "success",
                "analysis_result": result
            })
            
            # Rate limiting safety: Sleep 1.5s between patients (except after the very last one)
            if idx < len(files) - 1:
                await asyncio.sleep(1.5)
                
        except Exception as exc:
            logger.error(f"Error analyzing {file.filename} in batch: {exc}")
            batch_results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(exc)
            })
            
    return {
        "status": "completed",
        "total_files": len(files),
        "successful": sum(1 for r in batch_results if r["status"] == "success"),
        "results": batch_results
    }


@app.post("/baseline")
def run_baseline() -> Dict[str, Any]:
    try:
        from baseline import run_all_tasks
        scores = run_all_tasks()
        return {"baseline_scores": scores}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _build_detailed_baseline_report() -> Dict[str, Any]:
    """Run baseline on all tasks and return a detailed report per patient."""
    from baseline import _extract_search_terms_from_record, select_next_action
    from grader import evaluate_episode
    from models import ActionType

    report = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "patients": {}
    }

    for task_level in ["easy", "medium", "hard"]:
        env = ClinicalTrialEnv()
        observation = env.reset(task_level)

        patient_record = observation.patient_record
        expected_trial = env.expected_trial

        patient_summary = {
            "primary_condition": "",
            "demographics": {},
            "positive_facts": [],
            "exclusion_risks": [],
            "timeline_facts": [],
            "prior_therapies": [],
            "uncertainties": [],
            "recommended_search_terms": _extract_search_terms_from_record(patient_record),
        }

        searched_terms = []
        trial_reviews = {}
        action_log = []

        for step_num in range(15):
            if env.is_done:
                break
            try:
                action = select_next_action(
                    observation=observation,
                    patient_summary=patient_summary,
                    trial_reviews=trial_reviews,
                    searched_terms=searched_terms,
                )
            except Exception as exc:
                action_log.append({"step": step_num + 1, "error": str(exc)})
                break

            action_entry = {
                "step": step_num + 1,
                "action_type": action.action_type.value,
            }

            if action.action_type == ActionType.SEARCH_TRIALS and action.search_query:
                searched_terms.append(action.search_query)
                action_entry["search_query"] = action.search_query

            pending_tid = action.trial_id if action.action_type == ActionType.READ_CRITERIA else None
            if action.trial_id:
                action_entry["trial_id"] = action.trial_id

            prev_assigned = env.assigned_trial_id
            observation, reward, done, info = env.step(action)
            action_entry["reward"] = reward.value
            action_entry["feedback"] = observation.system_feedback

            if env.assigned_trial_id != prev_assigned:
                action_entry["change"] = f"Assigned trial changed: {prev_assigned} → {env.assigned_trial_id}"

            if pending_tid and observation.current_trial_criteria:
                exclusion_risks = patient_summary.get("exclusion_risks", [])
                criteria_text = observation.current_trial_criteria.lower()
                has_exclusion_hit = any(
                    risk.lower() in criteria_text for risk in exclusion_risks if risk
                )
                trial_reviews[pending_tid] = {
                    "trial_id": pending_tid,
                    "decision": "ineligible" if has_exclusion_hit else "eligible",
                    "exclude_hits": [r for r in exclusion_risks if r.lower() in criteria_text],
                }
                action_entry["trial_review"] = trial_reviews[pending_tid]

            action_log.append(action_entry)
            if done:
                break

        final_score = evaluate_episode(env.state(), env.expected_trial)
        report["patients"][task_level] = {
            "task_level": task_level,
            "patient_record": patient_record,
            "expected_trial": expected_trial,
            "final_assigned_trial": env.assigned_trial_id,
            "score": final_score,
            "total_steps": env.step_count,
            "trials_reviewed": trial_reviews,
            "action_log": action_log,
            "result": "CORRECT ✓" if env.assigned_trial_id == expected_trial else f"WRONG ✗ (expected {expected_trial}, got {env.assigned_trial_id})"
        }

    return report


@app.get("/baseline", response_class=HTMLResponse)
def baseline_ui() -> HTMLResponse:
    """Beautiful HTML dashboard showing baseline results with per-patient download."""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Baseline Results — Clinical Trial Matchmaker</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      min-height: 100vh;
      color: #e2e8f0;
      padding: 2rem 1rem;
    }
    h1 {
      text-align: center;
      font-size: 2.2rem;
      font-weight: 700;
      background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.4rem;
    }
    .subtitle {
      text-align: center;
      color: #94a3b8;
      font-size: 0.95rem;
      margin-bottom: 2.5rem;
    }
    .run-btn {
      display: block;
      margin: 0 auto 2.5rem;
      padding: 0.85rem 2.5rem;
      background: linear-gradient(135deg, #7c3aed, #2563eb);
      color: #fff;
      border: none;
      border-radius: 12px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.2s, box-shadow 0.2s;
      box-shadow: 0 4px 20px rgba(124,58,237,0.4);
    }
    .run-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(124,58,237,0.6); }
    .run-btn:active { transform: translateY(0); }
    .run-btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; max-width: 1100px; margin: 0 auto 3rem; }
    .card {
      background: rgba(255,255,255,0.05);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px;
      padding: 1.5rem;
      transition: transform 0.2s;
    }
    .card:hover { transform: translateY(-4px); }
    .card-header {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 1rem;
    }
    .badge {
      padding: 0.25rem 0.75rem;
      border-radius: 20px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .easy .badge { background: rgba(52,211,153,0.2); color: #34d399; }
    .medium .badge { background: rgba(251,191,36,0.2); color: #fbbf24; }
    .hard .badge { background: rgba(248,113,113,0.2); color: #f87171; }
    .score-ring {
      font-size: 2rem;
      font-weight: 700;
      text-align: center;
      margin: 0.75rem 0;
    }
    .score-ring.correct { color: #34d399; }
    .score-ring.wrong { color: #f87171; }
    .info-row {
      display: flex;
      justify-content: space-between;
      font-size: 0.82rem;
      color: #94a3b8;
      margin-top: 0.4rem;
    }
    .info-row span:last-child { color: #e2e8f0; font-weight: 500; }
    .record-box {
      background: rgba(0,0,0,0.3);
      border-radius: 8px;
      padding: 0.75rem;
      font-size: 0.78rem;
      color: #cbd5e1;
      margin: 0.8rem 0;
      line-height: 1.5;
      max-height: 80px;
      overflow: hidden;
      position: relative;
    }
    .record-box::after {
      content: '';
      position: absolute;
      bottom: 0; left: 0; right: 0;
      height: 30px;
      background: linear-gradient(transparent, rgba(0,0,0,0.6));
    }
    .dl-btn {
      display: block;
      width: 100%;
      padding: 0.6rem;
      margin-top: 1rem;
      background: rgba(99,102,241,0.2);
      border: 1px solid rgba(99,102,241,0.4);
      border-radius: 8px;
      color: #a5b4fc;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      text-align: center;
      transition: background 0.2s;
      text-decoration: none;
    }
    .dl-btn:hover { background: rgba(99,102,241,0.4); color: #fff; }
    .dl-all-btn {
      display: block;
      width: fit-content;
      margin: 0 auto 2rem;
      padding: 0.7rem 2rem;
      background: linear-gradient(135deg, #059669, #0891b2);
      color: #fff;
      border-radius: 10px;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      transition: transform 0.2s;
    }
    .dl-all-btn:hover { transform: translateY(-2px); }
    #status {
      text-align: center;
      color: #94a3b8;
      font-size: 0.9rem;
      margin-bottom: 1.5rem;
      min-height: 1.2rem;
    }
    .spinner {
      display: inline-block;
      width: 14px; height: 14px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: #a78bfa;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      vertical-align: middle;
      margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .hidden { display: none !important; }
    footer { text-align: center; color: #475569; font-size: 0.78rem; margin-top: 3rem; }
  </style>
</head>
<body>
  <h1>🏥 Baseline Results</h1>
  <p class="subtitle">Clinical Trial Matchmaker — Rule-Based Agent Performance</p>

  <button class="run-btn" id="runBtn" onclick="runBaseline()">▶ Run Baseline Now</button>
  <div id="status"></div>

  <div class="cards hidden" id="cards">
    <!-- Cards injected by JS -->
  </div>

  <a id="dlAllBtn" class="dl-all-btn hidden" href="/baseline/download" download="baseline_full_report.json">⬇ Download Full Report (All Patients)</a>

  <footer>Generated by Clinical Trial Matchmaker API</footer>

  <script>
    let results = null;

    async function runBaseline() {
      const btn = document.getElementById('runBtn');
      const status = document.getElementById('status');
      btn.disabled = true;
      status.innerHTML = '<span class="spinner"></span> Running baseline agent on all patients…';
      document.getElementById('cards').classList.add('hidden');
      document.getElementById('dlAllBtn').classList.add('hidden');

      try {
        const res = await fetch('/baseline', { method: 'POST' });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        results = data.baseline_scores;
        renderCards(results);
        status.innerHTML = '✅ Baseline complete! Scroll down to see results.';
        document.getElementById('dlAllBtn').classList.remove('hidden');
      } catch (e) {
        status.innerHTML = '❌ Error: ' + e.message;
      } finally {
        btn.disabled = false;
      }
    }

    function renderCards(scores) {
      const cards = document.getElementById('cards');
      cards.innerHTML = '';
      const levels = ['easy', 'medium', 'hard'];
      const emojis = { easy: '🟢', medium: '🟡', hard: '🔴' };
      levels.forEach(level => {
        const score = scores[level] ?? 'N/A';
        const isCorrect = score === 1.0;
        const card = document.createElement('div');
        card.className = 'card ' + level;
        card.innerHTML = `
          <div class="card-header">
            <span>${emojis[level]} ${level.charAt(0).toUpperCase() + level.slice(1)} Task</span>
            <span class="badge">${level}</span>
          </div>
          <div class="score-ring ${isCorrect ? 'correct' : 'wrong'}">
            ${isCorrect ? '✓ 1.0' : '✗ ' + score}
          </div>
          <div class="info-row"><span>Score</span><span>${score}</span></div>
          <div class="info-row"><span>Outcome</span><span>${isCorrect ? 'Correct match ✓' : 'Incorrect ✗'}</span></div>
          <a class="dl-btn" href="/baseline/report/${level}" download="patient_${level}_report.json">
            ⬇ Download ${level.charAt(0).toUpperCase() + level.slice(1)} Patient Report
          </a>
        `;
        cards.appendChild(card);
      });
      cards.classList.remove('hidden');
    }
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/baseline/download")
def download_full_baseline_report():
    """Download full baseline report for all patients as a JSON file."""
    try:
        report = _build_detailed_baseline_report()
        content = json.dumps(report, indent=2, ensure_ascii=False)
        filename = f"baseline_full_report_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/baseline/report/{task_level}")
def download_patient_report(task_level: str):
    """Download a detailed report for a specific patient (easy, medium, or hard)."""
    if task_level not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail="task_level must be easy, medium, or hard")
    try:
        report = _build_detailed_baseline_report()
        patient_data = report["patients"][task_level]
        content = json.dumps({
            "generated_at": report["generated_at"],
            "task_level": task_level,
            "patient": patient_data
        }, indent=2, ensure_ascii=False)
        filename = f"patient_{task_level}_report_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
