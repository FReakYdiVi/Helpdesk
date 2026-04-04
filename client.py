"""HTTP client for the Helpdesk OpenEnv server (see server/app.py)."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from .models import Action, Observation, Reward


@dataclass
class StepResult:
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class HelpdeskEnvClient:
    """Minimal client for POST /reset and POST /step on the FastAPI server."""

    def __init__(
        self,
        base_url: str,
        request_timeout_s: float = 60.0,
    ):
        self._base = base_url.rstrip("/")
        self._timeout = float(request_timeout_s)
        self._http = requests.Session()

    def reset(self, task_id: str = "easy") -> StepResult:
        r = self._http.post(
            f"{self._base}/reset",
            json={"task_id": task_id},
            timeout=self._timeout,
        )
        r.raise_for_status()
        data = r.json()
        obs = Observation(**data["observation"])
        rew = (
            Reward(**data["reward"])
            if data.get("reward") is not None
            else Reward(
                value=0.0,
                correctness=0.0,
                safety=1.0,
                resolution=0.0,
                efficiency=0.0,
                penalties=0.0,
                done=False,
                info={},
            )
        )
        return StepResult(
            observation=obs,
            reward=rew,
            done=bool(data.get("done", False)),
            info=dict(data.get("info") or {}),
        )

    def step(self, action: Action) -> StepResult:
        r = self._http.post(
            f"{self._base}/step",
            json={"action": action.model_dump()},
            timeout=self._timeout,
        )
        r.raise_for_status()
        data = r.json()
        return StepResult(
            observation=Observation(**data["observation"]),
            reward=Reward(**data["reward"]),
            done=bool(data.get("done", False)),
            info=dict(data.get("info") or {}),
        )

    def state(self) -> Observation:
        r = self._http.get(f"{self._base}/state", timeout=self._timeout)
        r.raise_for_status()
        data = r.json()
        return Observation(**data["observation"])

    def health(self) -> Dict[str, str]:
        r = self._http.get(f"{self._base}/health", timeout=self._timeout)
        r.raise_for_status()
        return dict(r.json())
