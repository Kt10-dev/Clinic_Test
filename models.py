from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import List, Dict, Optional, Any
from enum import Enum

# -----------------------------------------
# Action Space (Agent क्या-क्या कर सकता है)
# -----------------------------------------
class ActionType(str, Enum):
    SEARCH_TRIALS = "search_trials"              # बीमारी के नाम से trials ढूँढना
    READ_CRITERIA = "read_criteria"              # किसी specific trial की details/criteria पढ़ना
    ASSIGN_TRIAL = "assign_trial"                # मरीज़ को किसी trial के लिए assign करना
    MARK_INELIGIBLE = "mark_ineligible"          # अगर मरीज़ किसी भी trial के लायक नहीं है
    SUBMIT = "submit"                            # Task खत्म करना

class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType = Field(
        ..., 
        description="The type of action the agent wants to perform."
    )
    search_query: Optional[str] = Field(
        None, 
        description="Disease or condition to search for (e.g., 'Type 2 Diabetes'). Required for 'search_trials'."
    )
    trial_id: Optional[str] = Field(
        None, 
        description="The ID of the clinical trial. Required for 'read_criteria' and 'assign_trial'."
    )

    @model_validator(mode="after")
    def validate_required_fields(self) -> "Action":
        if self.action_type == ActionType.SEARCH_TRIALS and not self.search_query:
            raise ValueError("search_query is required for 'search_trials'.")

        if self.action_type in {
            ActionType.READ_CRITERIA,
            ActionType.ASSIGN_TRIAL,
        } and not self.trial_id:
            raise ValueError(
                "trial_id is required for 'read_criteria' and 'assign_trial'."
            )

        return self

# -----------------------------------------
# Observation Space (Agent को क्या दिखेगा)
# -----------------------------------------
class Observation(BaseModel):
    patient_record: str = Field(
        ..., 
        description="The electronic health record (EHR) of the patient, including age, conditions, and recent lab results."
    )
    search_results: Optional[List[Dict[str, str]]] = Field(
        None, 
        description="List of trials returned from the last 'search_trials' action. Contains trial_id and title."
    )
    current_trial_criteria: Optional[str] = Field(
        None, 
        description="Detailed Inclusion and Exclusion criteria of the trial queried using 'read_criteria'."
    )
    assigned_trial_id: Optional[str] = Field(
        None, 
        description="The trial ID currently assigned to the patient. Null if none assigned yet."
    )
    system_feedback: str = Field(
        ..., 
        description="Feedback from the environment (e.g., 'Found 3 trials', 'Trial CT-102 assigned', or Error messages)."
    )

# -----------------------------------------
# Reward Space (Agent को score कैसे मिलेगा)
# -----------------------------------------
class Reward(BaseModel):
    value: float = Field(
        ..., 
        description="Reward for the step. Negative for useless actions, partial positive for finding the right trial, 1.0 for correct submission."
    )
    done: bool = Field(
        ..., 
        description="True if the agent submitted the final answer, ending the episode."
    )
    info: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional info for grading (e.g., 'correct_trial_expected')."
    )
