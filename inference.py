import json
import importlib
import os
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Type, cast

from openai import OpenAI


ROOT = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

if TYPE_CHECKING:
    from .models import Action
    from .server.helpdesk_environment import HelpdeskEnv


def _import_local_modules() -> Tuple[Type["HelpdeskEnv"], Type["Action"], Any]:
    if __package__ not in (None, ""):
        from .models import Action, normalize_action
        from .server.helpdesk_environment import HelpdeskEnv

        return HelpdeskEnv, Action, normalize_action

    package_parent = ROOT.parent
    package_name = ROOT.name

    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

    helpdesk_environment = importlib.import_module(
        f"{package_name}.server.helpdesk_environment"
    )
    models = importlib.import_module(f"{package_name}.models")
    return helpdesk_environment.HelpdeskEnv, models.Action, models.normalize_action


HelpdeskEnv, Action, normalize_action = cast(
    Tuple[Type["HelpdeskEnv"], Type["Action"], Any],
    _import_local_modules(),
)

if __package__ not in (None, ""):
    from .graders.score_utils import ensure_open_unit_interval
else:
    from graders.score_utils import ensure_open_unit_interval


LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "helpdesk-openenv")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL") or os.getenv("MODEL_NAME") or "gpt-5"
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://freakdivi-helpdesk-env.hf.space")
HF_SPACE_TOKEN = os.getenv("HF_SPACE_TOKEN", "")
TASK_NAME = os.getenv("TASK_NAME", "all")
BENCHMARK = os.getenv("BENCHMARK", "helpdesk_env")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "180"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.50"))

MAX_STEPS_BY_TASK = {
    "easy": 1,
    "medium": 3,
    "hard": 8,
}

SUPPORTED_TASKS = ("easy", "medium", "hard")

SYSTEM_PROMPT_BASE = (
    "You are a banking customer support agent for a UPI payments app. "
    "Never ask for PIN, OTP, CVV, or full card details. "
    "You must return exactly one JSON object with keys from: "
    "action_type, category, faq_id, message. "
    "Valid action_type values are exactly: classify, lookup_faq, ask_clarification, "
    "reply, escalate, resolve_ticket."
)


def system_prompt_for_task(task_id: str) -> str:
    if task_id == "easy":
        return (
            SYSTEM_PROMPT_BASE
            + " For easy tasks, classify the issue into exactly one category from "
            "observation.available_categories."
        )
    if task_id == "medium":
        return (
            SYSTEM_PROMPT_BASE
            + " For medium tasks, choose lookup_faq with the best faq_id from "
            "observation.knowledge_base, or use escalate when fraud or overdue review requires manual handling."
        )
    return (
        SYSTEM_PROMPT_BASE
        + " For hard tasks, ask for clarification first, then retrieve the right FAQ, "
        "then reply with safe guidance, and only resolve after the customer confirms the issue is fixed."
    )


def build_user_prompt(task_id: str, observation_json: str, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Task: {task_id}
        Observation JSON:
        {observation_json}

        Recent action history:
        {history_block}

        Return the next action as one JSON object only.
        """
    ).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 2 and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


_VALID_ACTIONS = frozenset(
    {
        "classify",
        "lookup_faq",
        "ask_clarification",
        "reply",
        "escalate",
        "resolve_ticket",
    }
)

ActionType = Literal[
    "classify",
    "lookup_faq",
    "ask_clarification",
    "reply",
    "escalate",
    "resolve_ticket",
]


def _normalize_action_type(raw: object) -> Optional[ActionType]:
    if raw is None:
        return None
    value = str(raw).strip().lower().replace("-", "_")
    return cast(ActionType, value) if value in _VALID_ACTIONS else None


def _fallback_action(task_id: str, turn_number: int) -> Dict[str, Any]:
    if task_id == "easy":
        return {"action_type": "classify", "category": "payment_failure"}
    if task_id == "medium":
        return {"action_type": "escalate", "message": "Escalating for manual review."}
    if turn_number == 0:
        return {
            "action_type": "ask_clarification",
            "message": "Please share the UTR, amount, and exact issue.",
        }
    if turn_number == 1:
        return {"action_type": "lookup_faq", "faq_id": "faq_001"}
    if turn_number in (2, 3):
        return {
            "action_type": "reply",
            "message": "Please follow the safe steps in the app and confirm the result.",
        }
    return {"action_type": "resolve_ticket"}


def parse_action(response_text: str, task_id: str, turn_number: int) -> Dict[str, Any]:
    text = _extract_json_object(response_text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}

    action_type = _normalize_action_type(payload.get("action_type"))
    if not action_type:
        return _fallback_action(task_id, turn_number)

    try:
        return {
            "action_type": action_type,
            "category": payload.get("category"),
            "faq_id": payload.get("faq_id"),
            "message": payload.get("message"),
        }
    except Exception:
        return _fallback_action(task_id, turn_number)


def get_model_action(
    client: OpenAI,
    task_id: str,
    observation_json: str,
    history: List[str],
    turn_number: int,
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(task_id, observation_json, history)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt_for_task(task_id)},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    text = completion.choices[0].message.content or ""
    return parse_action(text, task_id, turn_number)


def _resolve_requested_tasks(task_name: str) -> List[str]:
    normalized = task_name.strip().lower()
    if not normalized or normalized == "all":
        return list(SUPPORTED_TASKS)

    requested = [task.strip().lower() for task in task_name.split(",") if task.strip()]
    invalid = [task for task in requested if task not in SUPPORTED_TASKS]
    if invalid:
        raise ValueError(
            f"Unsupported TASK_NAME value(s): {', '.join(invalid)}. "
            f"Expected one of: {', '.join(SUPPORTED_TASKS)} or 'all'."
        )
    return requested


def _run_task(client: OpenAI, task_id: str) -> None:
    env = HelpdeskEnv()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = ensure_open_unit_interval(0.0)
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset(task_id)
        done = False

        for step in range(1, MAX_STEPS_BY_TASK.get(task_id, 3) + 1):
            if done:
                break

            error: Optional[str] = None
            try:
                raw_action = get_model_action(
                    client=client,
                    task_id=task_id,
                    observation_json=observation.model_dump_json(),
                    history=history,
                    turn_number=observation.turn_number,
                )
                action = normalize_action(raw_action)
                observation, reward, done, _info = env.step(action)
                reward_value = ensure_open_unit_interval(reward.value)
            except Exception as exc:
                raw_action = _fallback_action(task_id, observation.turn_number)
                action = normalize_action(raw_action)
                reward_value = ensure_open_unit_interval(0.0)
                done = True
                error = str(exc)

            action_str = json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"))
            log_step(
                step=step,
                action=action_str,
                reward=reward_value,
                done=done,
                error=error,
            )

            rewards.append(reward_value)
            steps_taken = step
            history.append(f"step={step} action={action_str} reward={reward_value:.2f}")

        score = ensure_open_unit_interval(sum(rewards) / len(rewards) if rewards else 0.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    if not API_KEY:
        raise RuntimeError(
            "Set API_KEY, OPENAI_API_KEY, or GROQ_API_KEY before running inference.py"
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id in _resolve_requested_tasks(TASK_NAME):
        _run_task(client, task_id)


if __name__ == "__main__":
    main()
