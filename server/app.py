"""FastAPI server exposing HelpdeskEnv over HTTP."""

from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from .helpdesk_environment import HelpdeskEnv
from ..models import Action, Reward, normalize_action

app = FastAPI(title="Helpdesk OpenEnv")
_env: Optional[HelpdeskEnv] = None

TASKS: List[Dict[str, Any]] = [
    {
        "id": "easy",
        "difficulty": "easy",
        "description": "Classify the customer's issue into the correct support category.",
        "max_steps": 1,
        "grader": {
            "type": "llm",
            "prompt_template": (
                "Score the agent's performance for the easy helpdesk task on a scale "
                "from 0.001 to 0.999. Reward correct issue classification, safe "
                "behavior, and efficient completion. Penalize incorrect categories, "
                "unsafe requests for sensitive information, or invalid actions. "
                "Return only a numeric score."
            ),
        },
    },
    {
        "id": "medium",
        "difficulty": "medium",
        "description": "Select the correct FAQ or escalate cases that require manual handling.",
        "max_steps": 3,
        "grader": {
            "type": "llm",
            "prompt_template": (
                "Score the agent's performance for the medium helpdesk task on a scale "
                "from 0.001 to 0.999. Reward selecting the correct FAQ or making the "
                "correct escalation decision, while maintaining safe guidance and good "
                "efficiency. Penalize incorrect retrieval, missed escalation, unsafe "
                "behavior, or unnecessary extra steps. Return only a numeric score."
            ),
        },
    },
    {
        "id": "hard",
        "difficulty": "hard",
        "description": (
            "Run a multi-turn support conversation with clarification, guidance, "
            "and safe closure."
        ),
        "max_steps": 8,
        "grader": {
            "type": "llm",
            "prompt_template": (
                "Score the agent's performance for the hard helpdesk task on a scale "
                "from 0.001 to 0.999. Reward appropriate clarification, correct FAQ "
                "retrieval, safe and useful guidance, and closing the case only when "
                "the issue is actually resolved. Penalize unsafe behavior, premature "
                "closure, missing clarification, or poor multi-turn handling. Return "
                "only a numeric score."
            ),
        },
    },
]


def get_env() -> HelpdeskEnv:
    global _env
    if _env is None:
        _env = HelpdeskEnv()
    return _env


class ResetBody(BaseModel):
    task_id: Literal["easy", "medium", "hard"] = "easy"


def _zero_reward() -> Dict[str, Any]:
    return Reward(
        value=0.0,
        correctness=0.0,
        safety=1.0,
        resolution=0.0,
        efficiency=0.0,
        penalties=0.0,
        done=False,
        info={},
    ).model_dump()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "UPI Banking Support Environment",
        "status": "running",
        "endpoints": ["/health", "/metadata", "/tasks", "/reset", "/step", "/state"],
    }


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    return {
        "name": "helpdesk_env",
        "description": "UPI banking customer support environment with 3 graded tasks.",
        "task_count": len(TASKS),
        "tasks": TASKS,
    }


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    return {"tasks": TASKS}


@app.post("/reset")
def reset(body: ResetBody = ResetBody()) -> Dict[str, Any]:
    obs = get_env().reset(body.task_id)
    return {
        "observation": obs.model_dump(),
        "reward": _zero_reward(),
        "done": False,
        "info": {},
    }


@app.post("/step")
def step(body: Dict[str, Any]) -> Dict[str, Any]:
    action = normalize_action(body["action"])
    obs, reward, done, info = get_env().step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    obs = get_env().state()
    return {"observation": obs.model_dump()}


def main() -> None:
    uvicorn.run("helpdesk_env.server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
