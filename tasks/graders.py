""" 
Grading logic for the Safe Station Environment.
Complies with the Meta OpenEnv Hackathon 2026 validator requirements.
"""

def compute_score(total_reward: float, max_possible: float = 2000.0) -> float:
    """Official Meta leaderboard scoring logic (0.0 to 1.0)."""
    offset = 1000.0
    score = (total_reward + offset) / max_possible
    return max(0.0, min(1.0, score))

def grade_easy(session_results) -> float:
    """Grader for the Easy Start task."""
    # session_results is usually provided by the platform as a summary dict
    reward = session_results.get("total_reward", 0.0)
    return compute_score(reward, max_possible=1000.0)

def grade_medium(session_results) -> float:
    """Grader for the Medium Operations task."""
    reward = session_results.get("total_reward", 0.0)
    return compute_score(reward, max_possible=2000.0)

def grade_hard(session_results) -> float:
    """Grader for the Hard Constraints task."""
    reward = session_results.get("total_reward", 0.0)
    # Hard mode has higher potential but stricter penalties
    return compute_score(reward, max_possible=2500.0)
