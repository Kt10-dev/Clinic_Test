import random
import logging
from typing import Dict, Any, Tuple, List, Optional
from models import Action, ActionType, Observation, Reward

logger = logging.getLogger("clinical_trial_env")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ==========================================
# 1. Mock Database (Clinical Trials & Patients)
# ==========================================

# Hospital में चल रहे Clinical Trials का Data
TRIALS_DB = {
    "CT-101": {
        "title": "Efficacy of new Metformin variant for Type 2 Diabetes",
        "condition": "Diabetes",
        "criteria": "INCLUSION: Age >= 18, Type 2 Diabetes diagnosed. \nEXCLUSION: Type 1 Diabetes, Pregnant."
    },
    "CT-102": {
        "title": "Beta-blocker efficacy in severe Hypertension",
        "condition": "Hypertension",
        "criteria": "INCLUSION: Systolic BP > 150. \nEXCLUSION: History of Asthma, Heart Attack in last 6 months."
    },
    "CT-103": {
        "title": "Immunotherapy for Advanced Non-Small Cell Lung Cancer",
        "condition": "Lung Cancer",
        "criteria": "INCLUSION: Stage III or IV Lung Cancer, Failed at least one line of chemotherapy. \nEXCLUSION: Autoimmune disease, recent surgeries (last 30 days)."
    }
}

# Patients (Tasks: Easy, Medium, Hard)
PATIENTS_DB = {
    "easy": {
        "record": "Patient Age: 45. Gender: Male. History: Diagnosed with Type 2 Diabetes 3 years ago. Current medications: standard Metformin. No other major health issues.",
        "expected_trial": "CT-101"
    },
    "medium": {
        "record": "Patient Age: 62. Gender: Female. Vitals: Blood Pressure 160/95. History: Severe Hypertension. Note: Patient has a history of childhood Asthma but no recent attacks.",
        "expected_trial": "NONE" # (Asthma exclusion criteria will disqualify her from CT-102)
    },
    "hard": {
        "record": "Patient Age: 55. Gender: Male. Diagnosis: Stage IV Non-Small Cell Lung Cancer. Treatments: Underwent first-line chemotherapy 4 months ago (failed to respond). No autoimmune disorders. Surgery for appendix removal 3 years ago.",
        "expected_trial": "CT-103"
    }
}

class TrialRepository:
    """Abstracts database operations for Clinical Trials."""
    def __init__(self, db: Dict[str, Dict[str, str]] = None):
        # We inject the dictionary here to allow easy swapping to a real DB later
        self._db = db if db is not None else TRIALS_DB

    def search_by_query(self, query: str) -> List[Dict[str, str]]:
        if not query:
            return []
        query_lower = query.lower()
        results = []
        for tid, tdata in self._db.items():
            condition = tdata.get("condition", "").lower()
            title = tdata.get("title", "").lower()
            if query_lower in condition or query_lower in title:
                results.append({"trial_id": tid, "title": tdata.get("title", "Unknown Title")})
        return results

    def get_criteria(self, trial_id: str) -> Optional[str]:
        trial = self._db.get(trial_id)
        return trial.get("criteria") if trial else None

    def exists(self, trial_id: str) -> bool:
        return trial_id in self._db


# ==========================================
# 2. Environment Class
# ==========================================

class ClinicalTrialEnv:
    def __init__(self, repository: TrialRepository = None):
        self.repository = repository or TrialRepository()
        self.current_patient = ""
        self.expected_trial = None
        self.assigned_trial_id = None
        self.task_difficulty = "easy"
        self.step_count = 0
        self.max_steps = 10
        self.is_done = False
        self.last_search_results = None
        self.last_read_criteria = None
        self.total_reward = 0.0

    def reset(self, task_difficulty: str = "easy", custom_patient_record: str = None) -> Observation:
        """नया एपिसोड शुरू करता है। (Easy, Medium, या Hard task के हिसाब से मरीज़ सेट करता है)"""
        if custom_patient_record:
            self.task_difficulty = "custom"
            self.current_patient = custom_patient_record
            self.expected_trial = None
        else:
            if task_difficulty not in PATIENTS_DB:
                task_difficulty = "easy"
                
            self.task_difficulty = task_difficulty
            patient_data = PATIENTS_DB[task_difficulty]
            self.current_patient = patient_data["record"]
            self.expected_trial = patient_data.get("expected_trial")

        self.assigned_trial_id = None
        self.step_count = 0
        self.is_done = False
        self.last_search_results = None
        self.last_read_criteria = None
        self.total_reward = 0.0

        return self._get_observation(feedback=f"New patient record loaded ({self.task_difficulty} task). Please analyze and assign the correct trial.")

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Agent के action को execute करता है और State/Reward update करता है।"""
        if self.is_done:
            return self._get_observation("Episode is already done. Please reset."), Reward(value=0.0, done=True), True, {}

        self.step_count += 1
        feedback = ""
        step_reward = 0.0

        # 1. SEARCH_TRIALS Action
        if action.action_type == ActionType.SEARCH_TRIALS:
            query = action.search_query.lower() if action.search_query else ""
            results = self.repository.search_by_query(query)
            
            self.last_search_results = results
            if results:
                feedback = f"Found {len(results)} trials matching '{query}'."
                step_reward = 0.1  # Partial reward for finding something
            else:
                feedback = f"No trials found matching '{query}'."
                step_reward = -0.05 # Minor penalty for bad search
            logger.debug(f"Search trials for '{query}': returned {len(results)} results.")

        # 2. READ_CRITERIA Action
        elif action.action_type == ActionType.READ_CRITERIA:
            tid = action.trial_id
            criteria = self.repository.get_criteria(tid)
            if criteria:
                self.last_read_criteria = criteria
                feedback = f"Showing criteria for {tid}."
                step_reward = 0.1  # Partial reward for reading criteria
                logger.debug(f"Read criteria for trial {tid} successful.")
            else:
                feedback = f"Error: Trial ID '{tid}' not found."
                step_reward = -0.1
                logger.warning(f"Read criteria failed: Trial {tid} not found.")

        # 3. ASSIGN_TRIAL Action
        elif action.action_type == ActionType.ASSIGN_TRIAL:
            tid = action.trial_id
            if self.repository.exists(tid):
                self.assigned_trial_id = tid
                feedback = f"Patient tentatively assigned to trial {tid}. Submit to confirm."
            else:
                feedback = f"Error: Cannot assign. Trial ID '{tid}' not found."
                step_reward = -0.1

        # 4. MARK_INELIGIBLE Action
        elif action.action_type == ActionType.MARK_INELIGIBLE:
            self.assigned_trial_id = "NONE"
            feedback = "Patient marked as ineligible for any current trials. Submit to confirm."

        # 5. SUBMIT Action (Final Grading)
        elif action.action_type == ActionType.SUBMIT:
            self.is_done = True
            
            if self.assigned_trial_id == self.expected_trial:
                # Correct match!
                step_reward = 1.0
                feedback = "SUCCESS! Patient correctly routed."
                logger.info(f"Task completed successfully: Patient assigned to {self.assigned_trial_id}")
            else:
                # Wrong match
                step_reward = 0.0  # Final score is 0
                feedback = f"FAILED. Expected '{self.expected_trial}', but got '{self.assigned_trial_id}'."
                logger.info(f"Task failed: Assigned {self.assigned_trial_id}, Expected {self.expected_trial}")

        # Safety catch for infinite loops
        if self.step_count >= self.max_steps and not self.is_done:
            self.is_done = True
            feedback = "Maximum steps reached. Episode terminated."
            step_reward = 0.0
            logger.warning("Episode terminated due to max steps.")

        self.total_reward += step_reward
        
        # Ensure final reward is strictly mapped between 0.0 and 1.0 for the grader
        final_score = max(0.0, min(1.0, self.total_reward)) if self.is_done else step_reward

        obs = self._get_observation(feedback)
        rew = Reward(value=final_score, done=self.is_done, info={"expected": self.expected_trial})
        
        return obs, rew, self.is_done, {"step": self.step_count}

    def _get_observation(self, feedback: str) -> Observation:
        """Current state को Pydantic Observation model में pack करता है।"""
        return Observation(
            patient_record=self.current_patient,
            search_results=self.last_search_results,
            current_trial_criteria=self.last_read_criteria,
            assigned_trial_id=self.assigned_trial_id,
            system_feedback=feedback
        )

    def state(self) -> Dict[str, Any]:
        """Environment की internal state return करता है (Debugging के लिए)।"""
        return {
            "task_difficulty": self.task_difficulty,
            "step_count": self.step_count,
            "assigned_trial_id": self.assigned_trial_id,
            "is_done": self.is_done
        }