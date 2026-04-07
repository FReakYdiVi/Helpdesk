"""FastAPI server exposing HelpdeskEnv over HTTP."""

from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from .helpdesk_environment import HelpdeskEnv
from ..models import Action, Reward, normalize_action

app = FastAPI(title="Helpdesk OpenEnv")
_env: Optional[HelpdeskEnv] = None


def get_env() -> HelpdeskEnv:
    global _env
    if _env is None:
        _env = HelpdeskEnv()
    return _env


class ResetBody(BaseModel):
    task_id: str = "easy"


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
        "endpoints": ["/health", "/reset", "/step", "/state"],
    }


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
