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
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
if API_BASE_URL:
    API_BASE_URL = API_BASE_URL.strip().rstrip("/")
    if API_BASE_URL.endswith("/chat/completions"):
        API_BASE_URL = API_BASE_URL[:-17]

# HF_TOKEN is mandatory, no default allowed as per checklist
API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

if not API_KEY:
    logger.warning("HF_TOKEN not found in environment. Inference may fail.")

# Strict OpenAI Client as required by the rules
client = OpenAI(
    api_key=API_KEY or "dummy_key",
    base_url=API_BASE_URL if API_BASE_URL else None
)

def extract_json(text: str) -> Dict[str, Any]:
    """Robustly extracts JSON from a string, handling markdown blocks."""
    text = text.strip()
    try:
        # Handle cases where LLM wraps JSON in markdown blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except (ValueError, IndexError, json.JSONDecodeError):
        # Fallback to simple bracket finding if markdown stripping fails
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
        except:
            pass
    return {}

def complete_json(system_prompt: str, user_payload: Dict[str, Any], retries: int = 1) -> Dict[str, Any]:
    """Helper to fetch 0-temperature JSON evaluations with retries and error handling."""
    payload_text = json.dumps(user_payload, sort_keys=True, ensure_ascii=True)
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": payload_text},
    ]
    
    for attempt in range(retries + 1):
        try:
           
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0,
            )
            if not response.choices:
                logger.warning(f"Attempt {attempt+1}: LLM returned no choices.")
                continue
                
            content = response.choices[0].message.content
            if not content:
                logger.warning(f"Attempt {attempt+1}: LLM returned empty content.")
                continue

            parsed = extract_json(content)
            if parsed:
                return parsed
            logger.warning(f"Attempt {attempt+1}: LLM returned non-JSON content: {content[:100]}...")
        except Exception as e:
            logger.error(f"Attempt {attempt+1}: LLM request failed: {e}")
            if attempt == retries:
                break
    
    return {}

def build_patient_summary(patient_record: str) -> Dict[str, Any]:
    summary = complete_json(PATIENT_SUMMARY_PROMPT, {"patient_record": patient_record})
    primary_condition = str(summary.get("primary_condition") or "").strip()
    
    # Simple fallback: if LLM failed to give search terms, use condition or a default
    found_terms = _normalize_string_list(summary.get("recommended_search_terms"))
    if not found_terms:
        # Emergency heuristic: use primary condition or common disease keywords found in record
        if primary_condition:
            found_terms = [primary_condition]
        else:
            # Last resort: just use some generic terms (could be improved)
            found_terms = ["cancer", "diabetes", "hypertension"] 
            
    recommended_search_terms = found_terms
    
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

        try:
            observation, reward, done, info = env.step(action)
            # MANDATORY STEP LOGGING
            print(f"[STEP] step={env.step_count} reward={reward.value}", flush=True)
        except Exception as e:
            logger.error(f"Environment step failed: {e}")
            break

        if pending_review_trial_id and observation.current_trial_criteria and action.action_type == ActionType.READ_CRITERIA:
            trial_title = _candidate_title(observation.search_results, pending_review_trial_id)
            try:
                review = review_trial(
                    patient_summary=patient_summary,
                    trial_id=pending_review_trial_id,
                    trial_title=trial_title,
                    trial_criteria=observation.current_trial_criteria,
                )
                trial_reviews[pending_review_trial_id] = review
            except Exception as e:
                logger.error(f"Trial review failed for {pending_review_trial_id}: {e}")

    return evaluate_episode(env.state(), env.expected_trial)

if __name__ == "__main__":
    print(f"--- Executing OpenEnv Benchmark Inference ---")
    
    scores: Dict[str, float] = {}
    
    # Strictly evaluating the 3 mandated task difficulties
    for task_level in ["easy", "medium", "hard"]:
        # MANDATORY START LOGGING
        print(f"[START] task={task_level}", flush=True)
        env = None
        try:
            env = ClinicalTrialEnv()
            score = run_agent_on_task(env, task_level)
            scores[task_level] = score
            # MANDATORY END LOGGING
            print(f"[END] task={task_level} score={score} steps={env.step_count}", flush=True)
        except Exception as e:
            logger.critical(f"FATAL: Unhandled exception during task '{task_level}': {e}")
            scores[task_level] = 0.0
            steps = env.step_count if env else 0
            # MANDATORY END LOGGING (even on failure)
            print(f"[END] task={task_level} score=0.0 steps={steps}", flush=True)
    
    print(f"FINAL BENCHMARK SCORES: {json.dumps(scores, indent=2)}")
