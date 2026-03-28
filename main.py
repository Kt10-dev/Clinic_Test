import time
import asyncio
import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List
from uuid import uuid4

logger = logging.getLogger("clinical_trial_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import io

from environment import ClinicalTrialEnv, PATIENTS_DB
from grader import evaluate_episode
from models import Action

app = FastAPI(title="Clinical Trial Matchmaker Env")
@app.get("/")
def home():
    return {"message": "Clinical Trial Matchmaker API is Running!", "docs": "/docs"}

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


def _get_session_env(session_id: str) -> SessionRuntime:
    _cleanup_expired_sessions()
    
    with sessions_lock:
        runtime = env_sessions.get(session_id)

    if runtime is None:
        logger.warning(f"Session {session_id} not found or expired.")
        raise HTTPException(status_code=404, detail="Session not found.")

    runtime.last_accessed = time.time()
    return runtime


class ResetRequest(BaseModel):
    task_difficulty: str = "easy"

@app.post("/reset")
def reset_env(req: ResetRequest) -> Dict[str, Any]:
    _cleanup_expired_sessions()
    if req.task_difficulty not in PATIENTS_DB:
        logger.warning(f"Invalid task difficulty requested: {req.task_difficulty}")
        raise HTTPException(
            status_code=400,
            detail="Invalid task difficulty. Choose easy, medium, or hard.",
        )

    env = ClinicalTrialEnv()
    observation = env.reset(req.task_difficulty)
    session_id = str(uuid4())

    with sessions_lock:
        env_sessions[session_id] = SessionRuntime(env=env)
        
    logger.info(f"Started new session {session_id} with difficulty {req.task_difficulty}")

    return {
        "session_id": session_id,
        "observation": observation.model_dump(),
    }


@app.post("/step/{session_id}")
def step_env(session_id: str, action: Action) -> Dict[str, Any]:
    runtime = _get_session_env(session_id)
    with runtime.lock:
        observation, reward, done, info = runtime.env.step(action)

    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state/{session_id}")
def get_state(session_id: str) -> Dict[str, Any]:
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


@app.get("/grader/{session_id}")
def get_grader_score(session_id: str) -> Dict[str, float]:
    runtime = _get_session_env(session_id)
    with runtime.lock:
        score = evaluate_episode(runtime.env.state(), runtime.env.expected_trial)
    return {"score": score}


@app.delete("/session/{session_id}")
def close_session(session_id: str) -> Dict[str, str]:
    with sessions_lock:
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
