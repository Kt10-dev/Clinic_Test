"""
baseline.py
===========
Deterministic rule-based agent for Clinical Trial Matchmaker.
Provides prompts, helpers, action selection logic, and task runners.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("baseline")

# ==========================================
# LLM Prompts
# ==========================================

PATIENT_SUMMARY_PROMPT = """
You are a clinical data analyst. Given a raw patient EHR record, extract structured information.

Return a JSON object with these exact keys:
- primary_condition: string (main medical condition)
- demographics: object with age, gender keys (strings)
- positive_facts: list of strings (confirmed diagnoses, current treatments, test results that SUPPORT trial eligibility)
- exclusion_risks: list of strings (any factor that could DISQUALIFY patient from a trial - past conditions, contraindications)
- timeline_facts: list of strings (any date/time-sensitive facts, e.g. "surgery 3 years ago", "chemo 4 months ago")
- prior_therapies: list of strings (previous treatments or medications)
- uncertainties: list of strings (anything ambiguous or missing from the record)
- recommended_search_terms: list of strings (best search terms to find relevant trials in the database)

Be thorough about exclusion_risks — these are critical for patient safety.
""".strip()

TRIAL_REVIEW_PROMPT = """
You are a clinical trial eligibility specialist. Given a patient summary and trial criteria, determine eligibility.

Return a JSON object with these exact keys:
- decision: string, one of: "eligible", "ineligible", "needs_more_info"
- include_matches: list of strings (inclusion criteria the patient meets)
- exclude_hits: list of strings (exclusion criteria that disqualify the patient — be very strict)
- timeline_assessment: list of strings (analysis of any time-sensitive criteria)
- missing_information: list of strings (info needed but not available)
- rationale: string (brief explanation of the final decision)

IMPORTANT: If ANY exclusion criterion is met, the decision MUST be "ineligible".
Be extremely strict about exclusion criteria — patient safety is the priority.
""".strip()


# ==========================================
# Helper Functions
# ==========================================

def _normalize_string_list(value: Any) -> List[str]:
    """Normalize LLM output to a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def _candidate_title(search_results: Optional[List[Dict[str, str]]], trial_id: str) -> str:
    """Look up a trial title from the search results list."""
    if not search_results:
        return trial_id
    for result in search_results:
        if result.get("trial_id") == trial_id:
            return result.get("title", trial_id)
    return trial_id


# ==========================================
# Rule-Based Action Selection (No LLM needed)
# ==========================================

def select_next_action(
    observation,
    patient_summary: Dict[str, Any],
    trial_reviews: Dict[str, Dict[str, Any]],
    searched_terms: List[str],
):
    """
    Deterministic rule-based agent that selects the next action.
    Does NOT require an LLM — works offline for reproducibility.
    """
    from models import Action, ActionType

    search_results = observation.search_results or []

    # Step 1: Search for trials using recommended terms (if not yet searched)
    recommended_terms = patient_summary.get("recommended_search_terms", [])
    primary_condition = patient_summary.get("primary_condition", "")

    # Build list of terms we still need to search
    unsearched = [t for t in recommended_terms if t not in searched_terms]
    if not unsearched and primary_condition and primary_condition not in searched_terms:
        unsearched = [primary_condition]

    if unsearched:
        term = unsearched[0]
        return Action(action_type=ActionType.SEARCH_TRIALS, search_query=term)

    # Step 2: Read criteria for any trials not yet reviewed
    for trial in search_results:
        tid = trial.get("trial_id")
        if tid and tid not in trial_reviews:
            return Action(action_type=ActionType.READ_CRITERIA, trial_id=tid)

    # Step 3: Pick the best eligible trial
    eligible_trial_id = None
    for tid, review in trial_reviews.items():
        if review.get("decision") == "eligible" and not review.get("exclude_hits"):
            eligible_trial_id = tid
            break

    # Step 4: Assign or mark ineligible, then submit
    if eligible_trial_id:
        if observation.assigned_trial_id != eligible_trial_id:
            return Action(action_type=ActionType.ASSIGN_TRIAL, trial_id=eligible_trial_id)
    else:
        # No eligible trial found — check exclusion risks
        if observation.assigned_trial_id != "NONE":
            return Action(action_type=ActionType.MARK_INELIGIBLE)

    return Action(action_type=ActionType.SUBMIT)


# ==========================================
# High-Level Task Runners
# ==========================================

def run_agent_on_custom_record(env, record_text: str) -> Dict[str, Any]:
    """
    Run the deterministic agent on a custom patient record text.
    Used by /analyze-report and /analyze-batch endpoints.
    """
    from grader import evaluate_episode

    observation = env.reset(custom_patient_record=record_text)

    patient_summary = {
        "primary_condition": "",
        "demographics": {},
        "positive_facts": [],
        "exclusion_risks": [],
        "timeline_facts": [],
        "prior_therapies": [],
        "uncertainties": [],
        "recommended_search_terms": _extract_search_terms_from_record(record_text),
    }

    searched_terms: List[str] = []
    trial_reviews: Dict[str, Dict[str, Any]] = {}
    max_steps = 15

    for _ in range(max_steps):
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
            logger.error(f"Action selection failed: {exc}")
            break

        from models import ActionType
        if action.action_type == ActionType.SEARCH_TRIALS and action.search_query:
            searched_terms.append(action.search_query)

        pending_tid = action.trial_id if action.action_type == ActionType.READ_CRITERIA else None
        observation, reward, done, info = env.step(action)

        if pending_tid and observation.current_trial_criteria:
            trial_reviews[pending_tid] = {
                "trial_id": pending_tid,
                "trial_title": _candidate_title(observation.search_results, pending_tid),
                "decision": "eligible",  # Optimistic — no LLM to review
                "exclude_hits": [],
                "rationale": "Rule-based baseline assignment",
            }

        if done:
            break

    score = evaluate_episode(env.state(), env.expected_trial)
    return {
        "assigned_trial_id": env.assigned_trial_id,
        "score": score,
        "steps": env.step_count,
        "trials_reviewed": list(trial_reviews.keys()),
    }


def _extract_search_terms_from_record(record_text: str) -> List[str]:
    """Simple keyword extractor for patient records without an LLM."""
    conditions = [
        "Diabetes", "Hypertension", "Lung Cancer", "Breast Cancer",
        "Asthma", "Autoimmune", "Cancer", "Tumor",
    ]
    text_lower = record_text.lower()
    found = [c for c in conditions if c.lower() in text_lower]
    return found if found else ["general"]


def run_all_tasks() -> Dict[str, float]:
    """
    Run the deterministic baseline agent on all 3 task levels.
    Used by POST /baseline endpoint.
    """
    from environment import ClinicalTrialEnv
    from grader import evaluate_episode

    scores: Dict[str, float] = {}

    for task_level in ["easy", "medium", "hard"]:
        env = ClinicalTrialEnv()
        observation = env.reset(task_level)

        patient_summary = {
            "primary_condition": "",
            "demographics": {},
            "positive_facts": [],
            "exclusion_risks": [],
            "timeline_facts": [],
            "prior_therapies": [],
            "uncertainties": [],
            "recommended_search_terms": _extract_search_terms_from_record(
                observation.patient_record
            ),
        }

        searched_terms: List[str] = []
        trial_reviews: Dict[str, Dict[str, Any]] = {}

        for _ in range(15):
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
                logger.error(f"[{task_level}] Action error: {exc}")
                break

            from models import ActionType
            if action.action_type == ActionType.SEARCH_TRIALS and action.search_query:
                searched_terms.append(action.search_query)

            pending_tid = action.trial_id if action.action_type == ActionType.READ_CRITERIA else None
            observation, reward, done, info = env.step(action)

            if pending_tid and observation.current_trial_criteria:
                exclusion_risks = patient_summary.get("exclusion_risks", [])
                criteria_text = observation.current_trial_criteria.lower()

                # Simple rule: check if any exclusion risk appears in criteria
                has_exclusion_hit = any(
                    risk.lower() in criteria_text
                    for risk in exclusion_risks
                    if risk
                )
                trial_reviews[pending_tid] = {
                    "trial_id": pending_tid,
                    "trial_title": _candidate_title(observation.search_results, pending_tid),
                    "decision": "ineligible" if has_exclusion_hit else "eligible",
                    "exclude_hits": [r for r in exclusion_risks if r.lower() in criteria_text],
                    "rationale": "Rule-based evaluation",
                }

            if done:
                break

        score = evaluate_episode(env.state(), env.expected_trial)
        scores[task_level] = score
        logger.info(f"Baseline [{task_level}]: score={score}, assigned={env.assigned_trial_id}")

    return scores
