# grader.py
from typing import Dict, Any

def evaluate_episode(state: Dict[str, Any], expected_trial: str) -> float:
    """
    Grader function: 
    Returns 1.0 for exact match, 0.0 for wrong match or timeout.
    """
    if not state.get("is_done", False):
        return 0.0  # Episode complete नहीं हुआ
        
    assigned_trial = state.get("assigned_trial_id")
    
    if assigned_trial == expected_trial:
        return 1.0
    else:
        return 0.0