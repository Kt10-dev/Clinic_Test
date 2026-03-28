"""
Inference Script
===================================
MANDATORY HACKATHON COMPLIANCE FILE
Uses strictly HF_TOKEN, MODEL_NAME, API_BASE_URL via OpenAI Client.
"""

import os
import json
import logging
from typing import Dict, Any, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from pydantic import ValidationError

from environment import ClinicalTrialEnv
from grader import evaluate_episode
from models import Action, ActionType
from baseline import (
    PATIENT_SUMMARY_PROMPT, 
    TRIAL_REVIEW_PROMPT, 
    _candidate_title, 
    _normalize_string_list,
    select_next_action
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

# --- MANDATORY HACKATHON VARIABLES ---
API_BASE_URL = os.getenv("API_BASE_URL") # E.g., https://router.huggingface.co/v1
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

if not API_KEY:
    raise ValueError("HF_TOKEN or API_KEY must be set in the environment.")

# Strict OpenAI Client as required by the rules
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL if API_BASE_URL else None
)

def complete_json(system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to fetch 0-temperature JSON evaluations from the OpenAI-compatible endpoint."""
    payload_text = json.dumps(user_payload, sort_keys=True, ensure_ascii=True)
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": payload_text},
    ]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)

def build_patient_summary(patient_record: str) -> Dict[str, Any]:
    summary = complete_json(PATIENT_SUMMARY_PROMPT, {"patient_record": patient_record})
    primary_condition = str(summary.get("primary_condition") or "").strip()
    recommended_search_terms = _normalize_string_list(summary.get("recommended_search_terms"))
    
    if primary_condition and primary_condition not in recommended_search_terms:
        recommended_search_terms.insert(0, primary_condition)
        
    return {
        "primary_condition": primary_condition,
        "demographics": summary.get("demographics") or {},
        "positive_facts": _normalize_string_list(summary.get("positive_facts")),
        "exclusion_risks": _normalize_string_list(summary.get("exclusion_risks")),
        "timeline_facts": _normalize_string_list(summary.get("timeline_facts")),
        "prior_therapies": _normalize_string_list(summary.get("prior_therapies")),
        "uncertainties": _normalize_string_list(summary.get("uncertainties")),
        "recommended_search_terms": recommended_search_terms,
    }

def review_trial(patient_summary: Dict[str, Any], trial_id: str, trial_title: str, trial_criteria: str) -> Dict[str, Any]:
    review = complete_json(
        TRIAL_REVIEW_PROMPT,
        {
            "patient_summary": patient_summary,
            "trial_id": trial_id,
            "trial_title": trial_title,
            "trial_criteria": trial_criteria,
        }
    )
    decision = str(review.get("decision") or "").strip().lower()
    if decision not in {"eligible", "ineligible", "needs_more_info"}:
        decision = "needs_more_info"
        
    return {
        "trial_id": trial_id,
        "trial_title": trial_title,
        "decision": decision,
        "include_matches": _normalize_string_list(review.get("include_matches")),
        "exclude_hits": _normalize_string_list(review.get("exclude_hits")),
        "timeline_assessment": _normalize_string_list(review.get("timeline_assessment")),
        "missing_information": _normalize_string_list(review.get("missing_information")),
        "rationale": str(review.get("rationale") or "").strip(),
    }

def run_agent_on_task(env: ClinicalTrialEnv, task_level: str) -> float:
    """Executes a single end-to-end task (easy/medium/hard) returning a robust score 0.0-1.0"""
    observation = env.reset(task_level)
    patient_summary = build_patient_summary(observation.patient_record)

    searched_terms: List[str] = []
    trial_reviews: Dict[str, Dict[str, Any]] = {}

    while not env.is_done:
        try:
            action = select_next_action(
                observation=observation,
                patient_summary=patient_summary,
                trial_reviews=trial_reviews,
                searched_terms=searched_terms,
            )
        except ValidationError as exc:
            logger.error(f"Action validation failed: {exc}")
            break

        if action.action_type == ActionType.SEARCH_TRIALS and action.search_query:
            searched_terms.append(action.search_query)

        pending_review_trial_id = action.trial_id if action.action_type == ActionType.READ_CRITERIA else None

        observation, reward, done, info = env.step(action)

        if pending_review_trial_id and observation.current_trial_criteria and action.action_type == ActionType.READ_CRITERIA:
            trial_title = _candidate_title(observation.search_results, pending_review_trial_id)
            trial_reviews[pending_review_trial_id] = review_trial(
                patient_summary=patient_summary,
                trial_id=pending_review_trial_id,
                trial_title=trial_title,
                trial_criteria=observation.current_trial_criteria,
            )

    return evaluate_episode(env.state(), env.expected_trial)

if __name__ == "__main__":
    logger.info(f"--- Executing OpenEnv Benchmark Inference ---")
    logger.info(f"Model: {MODEL_NAME}")
    
    scores: Dict[str, float] = {}
    
    # Strictly evaluating the 3 mandated task difficulties
    for task_level in ["easy", "medium", "hard"]:
        env = ClinicalTrialEnv()
        score = run_agent_on_task(env, task_level)
        scores[task_level] = score
        logger.info(f"Task '{task_level}' completed. Score: {score}")
    
    logger.info(f"FINAL BENCHMARK SCORES: {json.dumps(scores, indent=2)}")
