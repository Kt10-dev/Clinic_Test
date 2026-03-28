import hashlib
import json
import logging
import os
import time

logger = logging.getLogger("clinical_trial_baseline")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI
from pydantic import ValidationError

from environment import ClinicalTrialEnv
from grader import evaluate_episode
from models import Action, ActionType

load_dotenv()

PATIENT_SUMMARY_PROMPT = """
You are a medical record normalizer for clinical trial matching.

Return a JSON object with this exact schema:
{
  "primary_condition": "string",
  "demographics": {
    "age": "string or null",
    "sex": "string or null"
  },
  "positive_facts": ["supported fact"],
  "exclusion_risks": ["risk or contraindication"],
  "timeline_facts": ["time-sensitive fact"],
  "prior_therapies": ["treatment history"],
  "uncertainties": ["missing but relevant fact"],
  "recommended_search_terms": ["broad database-friendly condition term"]
}

Rules:
- Use only facts explicitly supported by the patient record.
- Preserve negations and time windows literally.
- Normalize search terms to broad disease labels that a trial database is likely to index.
- Prefer terms such as "Lung Cancer" over overly specific staging phrases for search.
- Do not invent lab values, diagnoses, or medications.
"""

TRIAL_REVIEW_PROMPT = """
You are reviewing patient eligibility for one clinical trial.

Return a JSON object with this exact schema:
{
  "decision": "eligible" | "ineligible" | "needs_more_info",
  "include_matches": ["criterion satisfied by the patient"],
  "exclude_hits": ["criterion violated by the patient"],
  "timeline_assessment": ["time-based interpretation"],
  "missing_information": ["important unknown"],
  "rationale": "short explanation"
}

Rules:
- Exclusion criteria always override inclusion criteria.
- Evaluate temporal language literally.
- If a criterion is not proven from the record, mark it as missing instead of assuming it.
- Be conservative for high-risk medical decisions.
"""


class JsonLLMClient:
    def __init__(
        self,
        providers: List["ProviderConfig"],
        max_retries: int = 3,
    ) -> None:
        if not providers:
            raise ValueError("At least one provider configuration is required.")

        self.providers = providers
        self.provider_index = 0
        self.client = providers[0].client
        self.model = providers[0].model
        self.provider_name = providers[0].name
        self.max_retries = max_retries
        self.cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def from_env(cls) -> "JsonLLMClient":
        preferred_provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        groq_api_key = os.getenv("GROQ_API_KEY")

        provider_map: Dict[str, ProviderConfig] = {}
        if groq_api_key:
            provider_map["groq"] = ProviderConfig(
                name="groq",
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                client=OpenAI(
                    api_key=groq_api_key,
                    base_url="https://api.groq.com/openai/v1",
                ),
            )

        if openai_api_key:
            provider_map["openai"] = ProviderConfig(
                name="openai",
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                client=OpenAI(api_key=openai_api_key),
            )

        if not provider_map:
            raise ValueError(
                "Set OPENAI_API_KEY or GROQ_API_KEY in your environment before running the baseline."
            )

        if preferred_provider not in {"auto", "groq", "openai"}:
            raise ValueError(
                "LLM_PROVIDER must be one of: auto, groq, openai."
            )

        providers: List[ProviderConfig] = []
        if preferred_provider == "auto":
            for provider_name in ["groq", "openai"]:
                provider = provider_map.get(provider_name)
                if provider:
                    providers.append(provider)
        else:
            preferred = provider_map.get(preferred_provider)
            if preferred is None:
                raise ValueError(
                    f"LLM_PROVIDER='{preferred_provider}' is set but no matching API key was found."
                )
            providers.append(preferred)
            for provider_name, provider in provider_map.items():
                if provider_name != preferred_provider:
                    providers.append(provider)

        return cls(providers=providers)

    def _switch_provider(self) -> bool:
        if self.provider_index + 1 >= len(self.providers):
            return False

        self.provider_index += 1
        next_provider = self.providers[self.provider_index]
        self.client = next_provider.client
        self.model = next_provider.model
        self.provider_name = next_provider.name
        logger.warning(
            f"Authentication failed for the previous provider. Falling back to {self.provider_name}."
        )
        return True

    def complete_json(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
        cache_namespace: str,
    ) -> Dict[str, Any]:
        payload_text = json.dumps(user_payload, sort_keys=True, ensure_ascii=True)
        cache_key = hashlib.sha256(
            f"{cache_namespace}|{self.model}|{payload_text}".encode("utf-8")
        ).hexdigest()

        if cache_key in self.cache:
            return self.cache[cache_key]

        last_error: Optional[Exception] = None
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": payload_text},
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                parsed = json.loads(response.choices[0].message.content)
                self.cache[cache_key] = parsed
                return parsed
            except AuthenticationError as exc:
                last_error = exc
                if self._switch_provider():
                    continue
                break
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(attempt)

        raise RuntimeError(f"LLM JSON completion failed: {last_error}") from last_error


@dataclass
class ProviderConfig:
    name: str
    model: str
    client: OpenAI


def _normalize_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []

    normalized: List[str] = []
    for item in value:
        item_text = str(item).strip()
        if item_text:
            normalized.append(item_text)
    return normalized


def build_patient_summary(llm: JsonLLMClient, patient_record: str) -> Dict[str, Any]:
    summary = llm.complete_json(
        system_prompt=PATIENT_SUMMARY_PROMPT,
        user_payload={"patient_record": patient_record},
        cache_namespace="patient-summary",
    )

    primary_condition = str(summary.get("primary_condition") or "").strip()
    recommended_search_terms = _normalize_string_list(
        summary.get("recommended_search_terms")
    )

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


def review_trial(
    llm: JsonLLMClient,
    patient_summary: Dict[str, Any],
    trial_id: str,
    trial_title: str,
    trial_criteria: str,
) -> Dict[str, Any]:
    review = llm.complete_json(
        system_prompt=TRIAL_REVIEW_PROMPT,
        user_payload={
            "patient_summary": patient_summary,
            "trial_id": trial_id,
            "trial_title": trial_title,
            "trial_criteria": trial_criteria,
        },
        cache_namespace=f"trial-review::{trial_id}",
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
        "timeline_assessment": _normalize_string_list(
            review.get("timeline_assessment")
        ),
        "missing_information": _normalize_string_list(
            review.get("missing_information")
        ),
        "rationale": str(review.get("rationale") or "").strip(),
    }


def _candidate_title(search_results: Optional[List[Dict[str, str]]], trial_id: str) -> str:
    for trial in search_results or []:
        if trial.get("trial_id") == trial_id:
            return str(trial.get("title") or "")
    return ""


def select_next_action(
    observation,
    patient_summary: Dict[str, Any],
    trial_reviews: Dict[str, Dict[str, Any]],
    searched_terms: List[str],
) -> Action:
    if observation.assigned_trial_id:
        return Action(action_type=ActionType.SUBMIT)

    if observation.search_results is None:
        for term in patient_summary.get("recommended_search_terms", []):
            cleaned_term = term.strip()
            if cleaned_term and cleaned_term not in searched_terms:
                return Action(
                    action_type=ActionType.SEARCH_TRIALS,
                    search_query=cleaned_term,
                )
        fallback_term = patient_summary.get("primary_condition") or "clinical trial"
        return Action(action_type=ActionType.SEARCH_TRIALS, search_query=fallback_term)

    if observation.search_results:
        for trial in observation.search_results:
            trial_id = trial["trial_id"]
            if trial_id not in trial_reviews:
                return Action(action_type=ActionType.READ_CRITERIA, trial_id=trial_id)

        eligible_trials = [
            review for review in trial_reviews.values() if review["decision"] == "eligible"
        ]
        if eligible_trials:
            eligible_trials.sort(
                key=lambda review: (
                    len(review["exclude_hits"]) == 0,
                    len(review["include_matches"]),
                    -len(review["missing_information"]),
                ),
                reverse=True,
            )
            return Action(
                action_type=ActionType.ASSIGN_TRIAL,
                trial_id=eligible_trials[0]["trial_id"],
            )

    return Action(action_type=ActionType.MARK_INELIGIBLE)


def run_agent_on_task(
    env: ClinicalTrialEnv,
    task_level: str,
    llm: Optional[JsonLLMClient] = None,
) -> float:
    llm = llm or JsonLLMClient.from_env()
    observation = env.reset(task_level)
    patient_summary = build_patient_summary(llm, observation.patient_record)

    searched_terms: List[str] = []
    trial_reviews: Dict[str, Dict[str, Any]] = {}
    pending_review_trial_id: Optional[str] = None

    logger.info(f"\n--- Starting Task: {task_level.upper()} ---")
    logger.debug(
        f"Patient summary: {json.dumps(patient_summary, indent=2, ensure_ascii=True)}"
    )

    while not env.is_done:
        try:
            action = select_next_action(
                observation=observation,
                patient_summary=patient_summary,
                trial_reviews=trial_reviews,
                searched_terms=searched_terms,
            )
        except ValidationError as exc:
            logger.error(f"Action validation failed before stepping the environment: {exc}")
            break

        if action.action_type == ActionType.SEARCH_TRIALS and action.search_query:
            searched_terms.append(action.search_query)

        if action.action_type == ActionType.READ_CRITERIA:
            pending_review_trial_id = action.trial_id

        logger.info(
            f"Agent Action: {action.action_type} | "
            f"Target: {action.trial_id or action.search_query}"
        )

        observation, reward, done, info = env.step(action)
        logger.info(
            f"Reward: {reward.value} | Done: {done} | Feedback: {observation.system_feedback}"
        )

        if (
            pending_review_trial_id
            and observation.current_trial_criteria
            and action.action_type == ActionType.READ_CRITERIA
        ):
            trial_title = _candidate_title(
                observation.search_results,
                pending_review_trial_id,
            )
            trial_review = review_trial(
                llm=llm,
                patient_summary=patient_summary,
                trial_id=pending_review_trial_id,
                trial_title=trial_title,
                trial_criteria=observation.current_trial_criteria,
            )
            trial_reviews[pending_review_trial_id] = trial_review
            logger.debug(
                f"Eligibility review for {pending_review_trial_id}: {json.dumps(trial_review, indent=2, ensure_ascii=True)}"
            )
            pending_review_trial_id = None

    final_score = evaluate_episode(env.state(), env.expected_trial)
    logger.info(f"Task {task_level} Complete. Final Score: {final_score}")
    return final_score


def run_all_tasks() -> Dict[str, float]:
    llm = JsonLLMClient.from_env()
    scores: Dict[str, float] = {}

    for task_level in ["easy", "medium", "hard"]:
        env = ClinicalTrialEnv()
        scores[task_level] = run_agent_on_task(env, task_level, llm=llm)

    return scores


def run_agent_on_custom_record(
    env: ClinicalTrialEnv,
    record_text: str,
    llm: Optional[JsonLLMClient] = None,
) -> Dict[str, Any]:
    llm = llm or JsonLLMClient.from_env()
    observation = env.reset(custom_patient_record=record_text)
    patient_summary = build_patient_summary(llm, record_text)

    searched_terms: List[str] = []
    trial_reviews: Dict[str, Dict[str, Any]] = {}
    pending_review_trial_id: Optional[str] = None

    logger.info("\n--- Starting Custom Report Task ---")

    while not env.is_done:
        try:
            action = select_next_action(
                observation=observation,
                patient_summary=patient_summary,
                trial_reviews=trial_reviews,
                searched_terms=searched_terms,
            )
        except ValidationError as exc:
            logger.error(f"Action validation failed before stepping the environment: {exc}")
            break

        if action.action_type == ActionType.SEARCH_TRIALS and action.search_query:
            searched_terms.append(action.search_query)

        if action.action_type == ActionType.READ_CRITERIA:
            pending_review_trial_id = action.trial_id

        observation, reward, done, info = env.step(action)

        if (
            pending_review_trial_id
            and observation.current_trial_criteria
            and action.action_type == ActionType.READ_CRITERIA
        ):
            trial_title = _candidate_title(
                observation.search_results,
                pending_review_trial_id,
            )
            trial_review = review_trial(
                llm=llm,
                patient_summary=patient_summary,
                trial_id=pending_review_trial_id,
                trial_title=trial_title,
                trial_criteria=observation.current_trial_criteria,
            )
            trial_reviews[pending_review_trial_id] = trial_review
            pending_review_trial_id = None

    final_assignment = env.assigned_trial_id
    assigned_review = trial_reviews.get(final_assignment, {})
    
    return {
        "assigned_trial_id": final_assignment,
        "rationale": assigned_review.get("rationale", "No specific rationale was provided or patient is ineligible."),
        "patient_summary": patient_summary,
        "trial_reviews": trial_reviews
    }


if __name__ == "__main__":
    logger.info("Running staged clinical trial baseline...")
    scores = run_all_tasks()
    logger.info(f"Final Scores: {json.dumps(scores, indent=2)}")
