GOAL_BINDING = "reset-public-goal@2"
_MARKER = "Your task is to:"


def reset_public_goal(observation: str) -> str:
    if not isinstance(observation, str) or observation.count(_MARKER) != 1:
        raise ValueError("ALFWorld reset needs exactly one public task goal")
    goals = [
        line.strip()[len(_MARKER) :].strip()
        for line in observation.splitlines()
        if line.strip().startswith(_MARKER)
    ]
    if len(goals) != 1 or not goals[0] or "\x00" in goals[0]:
        raise ValueError("ALFWorld public task goal is missing or ambiguous")
    return goals[0]
