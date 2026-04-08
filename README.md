---
title: UPI Banking Support Environment
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - banking
  - upi
  - customer-support
---

# UPI Banking Support Environment

OpenEnv-style environment for evaluating agents on UPI customer support workflows. The benchmark focuses on realistic banking support decisions rather than generic FAQ matching.

## Motivation

This environment is designed to test whether an agent can behave like a safe and useful support assistant for a UPI payments product such as Paytm, PhonePe, or Google Pay style support flows.

The goal is not only to answer customers correctly, but also to:
- identify the right issue type
- retrieve the right knowledge entry
- escalate fraud or overdue review cases when needed
- avoid unsafe behavior such as asking for PINs or OTPs
- handle multi-turn conversations before closing a case

## Environment Description

The environment uses three tasks with increasing difficulty:
- `easy`: classify a customer issue into the correct support track
- `medium`: choose the right FAQ or escalate when human/manual review is required
- `hard`: run a short multi-turn support conversation with clarification, guidance, and closure

The current support tracks are:
- `payment_failure`
- `refund_delay`
- `fraud_complaint`
- `kyc_account_restriction`
- `upi_pin_or_bank_linking`

The dataset includes:
- 40 banking FAQ entries in [data/knowledge_base.json](data/knowledge_base.json)
- 10 `easy` tickets in [data/tickets/easy.json](data/tickets/easy.json)
- 10 `medium` tickets in [data/tickets/medium.json](data/tickets/medium.json)
- 10 `hard` tickets in [data/tickets/hard.json](data/tickets/hard.json)

## Architecture

The environment is organized around a small set of components that work together during each episode:

```text
Ticket datasets + knowledge base
data/tickets/*.json + data/knowledge_base.json
                |
                v
HelpdeskEnv
server/helpdesk_environment.py
reset() / step() / state()
                |
                +------------------------+
                |                        |
                v                        v
Observation for agent         UserSimulator for hard tasks
models.py                     user_simulator.py
                |                        |
                +-----------+------------+
                            |
                            v
Inference agent loop
inference.py
model call -> action -> normalized action
                            |
                            v
Graders + reward shaping
graders/*.py + reward logic in HelpdeskEnv
correctness / safety / resolution / efficiency / penalties
                            |
                            v
Per-step reward + final episode score
```

How it works in practice:

1. A task split such as `easy`, `medium`, or `hard` selects a ticket from the dataset.
2. [server/helpdesk_environment.py](server/helpdesk_environment.py) builds the observation and tracks hidden gold state.
3. [inference.py](inference.py) sends a compact task-specific prompt to the model and converts the reply into a valid action.
4. The environment applies the action, updates the conversation state, and calls the relevant grading logic.
5. For `hard` tasks, [user_simulator.py](user_simulator.py) produces realistic follow-up user responses.
6. The environment returns a shaped reward, and `inference.py` aggregates the episode score by task type.

## Why JSON Knowledge Base

We use a JSON knowledge base instead of an LLM-backed knowledge source for a few practical reasons:

- It gives a fixed source of truth for FAQ retrieval and evaluation.
- It makes matching the predicted FAQ against the gold FAQ simple and deterministic.
- It is easier to audit, expand, and maintain across categories and tags.
- It avoids adding extra model variance, latency, and cost inside the environment itself.
- It keeps retrieval quality and safe guidance grounded in controlled support content.

LLMs are still useful on top of the JSON knowledge base for response generation or judging nuanced outputs, but the knowledge source itself is intentionally structured and deterministic.

## Action Space

The public inference script and server accept the legacy action names below, which are internally mapped to the compact action model in [models.py](models.py).

| Action | Parameters | Purpose |
|---|---|---|
| `classify` | `category` | Predict the correct support track for an `easy` ticket |
| `lookup_faq` | `faq_id` | Choose the best FAQ entry for `medium` or `hard` |
| `ask_clarification` | `message` | Ask a question to gather missing details in `hard` |
| `reply` | `message` | Provide safe support guidance to the user |
| `escalate` | `message` | Escalate a case that should not be fully handled automatically |
| `resolve_ticket` | none | Close the case when it appears correctly resolved |

Internally, these are normalized to:
- `ask_for_details`
- `take_action`
- `respond_to_user`
- `escalate_case`
- `close_case`

## Observation Space

The model receives an `Observation` object from [models.py](models.py).

| Field | Type | Description |
|---|---|---|
| `case_id` | `str` | Unique identifier for the active ticket |
| `track` | `str` | Task split only: `easy`, `medium`, or `hard` |
| `customer_message` | `str` | Current customer issue text shown to the agent |
| `conversation_history` | `list[dict]` | Prior user/agent turns |
| `known_facts` | `dict` | Agent-visible state such as FAQ set, available categories, and progress flags |
| `required_slots` | `list[str]` | High-level missing information requirements for the episode |
| `available_actions` | `list[str]` | Actions allowed by the environment |
| `turn_number` | `int` | Current turn count |

Important evaluation detail:
- hidden gold labels such as the correct FAQ id and escalation label are not exposed to the model in the observation

## Reward

Rewards are normalized to the range `0.0` to `1.0` in [server/helpdesk_environment.py](server/helpdesk_environment.py).

The final reward is shaped rather than purely binary. It combines:
- `correctness`
- `safety`
- `resolution`
- `efficiency`
- `penalties`

Weighted reward:

```text
0.35 * correctness
+ 0.30 * safety
+ 0.20 * resolution
+ 0.15 * efficiency
+ penalties
```

Examples:
- correct classification gives a strong `easy` reward
- correct FAQ retrieval gives partial progress on `medium`
- correct escalation gives reward on `medium`
- clarification plus guidance plus successful closure raises `hard` reward
- unsafe prompts such as asking for PIN or OTP reduce reward sharply

## Inference Scoring

`inference.py` aggregates episode rewards differently for each task:

- `easy`: single-step scoring. Final score is the reward from the one step taken.
- `medium`: terminal-step scoring. Intermediate rewards are logged, but the final score is the reward from the last step taken.
- `hard`: discounted scoring. Final score is the normalized discounted cumulative reward:

```text
score = sum(reward_t * gamma^t for t, reward_t in enumerate(rewards))
        / sum(gamma^t for t in range(len(rewards)))
```

with `gamma = 0.9`.

## Task Difficulty

| Task | Difficulty | Description | Expected Agent Behavior |
|---|---|---|---|
| `easy` | Low | Single-turn issue classification | Identify the correct banking support track |
| `medium` | Medium | FAQ retrieval or escalation decision | Select the right FAQ or escalate fraud / overdue review cases |
| `hard` | High | Multi-turn support conversation | Ask clarification, guide safely, and close only when appropriate |

## Setup

From the package root:

```bash
cd /path/to/helpdesk_env
uv sync
```

Runtime configuration is read from `.env`.
The environment currently uses:
- `API_BASE_URL` for the provider endpoint
- `MODEL` or `MODEL_NAME` for the selected model
- `API_KEY` as the primary model credential
- `OPENAI_API_KEY` and `GROQ_API_KEY` are also supported as compatibility aliases
- `HF_SPACE_URL` for the deployed Space runtime URL
- `HF_SPACE_TOKEN` for protected Space access when required

## Usage

### Using Docker

```bash
# Build the image from the repository root
docker build -t helpdesk-openenv:latest .

# Run the server
docker run -p 8000:8000 helpdesk-openenv:latest
```

Docker smoke test:

```bash
curl http://127.0.0.1:8000/health

curl http://127.0.0.1:8000/

curl -X POST http://127.0.0.1:8000/reset \
  -H "Content-Type: application/json" \
  -d '{}'

curl -X POST http://127.0.0.1:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"action_type":"classify","category":"payment_failure"}}'

curl http://127.0.0.1:8000/state
```

### Local Development

```bash
# Quick compile check
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  inference.py server/app.py server/helpdesk_environment.py

# Run the server locally
uv run server
```

`uv run server` smoke test:

```bash
curl http://127.0.0.1:8000/health

curl http://127.0.0.1:8000/

curl -X POST http://127.0.0.1:8000/reset \
  -H "Content-Type: application/json" \
  -d '{}'

curl -X POST http://127.0.0.1:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"action_type":"classify","category":"payment_failure"}}'

curl http://127.0.0.1:8000/state
```

### Run Inference

```bash
API_BASE_URL=https://api.openai.com/v1 \
API_KEY=$OPENAI_API_KEY \
MODEL=gpt-5 \
TASK_NAME=easy \
python3 inference.py
```

```bash
API_BASE_URL=https://api.groq.com/openai/v1 \
API_KEY=$GROQ_API_KEY \
MODEL=llama-3.3-70b-versatile \
TASK_NAME=easy \
python3 inference.py
```

`inference.py` reads configuration from `.env`.

The script prints structured logs in the required format:

```text
[START] task=easy env=helpdesk_env model=llama-3.3-70b-versatile
[STEP] step=1 action={"action_type":"classify","category":"payment_failure"} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.000 rewards=1.00
```

### Use the Python Client

```python
from helpdesk_env.client import HelpdeskEnvClient

client = HelpdeskEnvClient("http://127.0.0.1:8000")
result = client.reset("easy")
print(result.observation.customer_message)
```

For a deployed HF Space:

```python
from helpdesk_env.client import HelpdeskEnvClient

client = HelpdeskEnvClient.from_env()
print(client.health())
```

### Test the Live HF Space

```bash

curl -X POST "https://freakdivi-helpdesk.hf.space/reset" \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy"}'

curl -X POST "https://freakdivi-helpdesk.hf.space/step" \
  -H "Content-Type: application/json" \
  -d '{"action":{"action_type":"classify","category":"payment_failure"}}'
```

## Hugging Face Space Deployment

This repo is configured as a Docker-based HF Space through the YAML frontmatter at the top of this README:
- `sdk: docker`
- `app_port: 8000`
- `tags` include `openenv`

Live Space:
- https://huggingface.co/spaces/Freakdivi/helpdesk_env


## Baseline Scores

Latest observed Groq baseline run after removing answer leakage from the observation:

| Model | Easy | Medium | Hard |
|---|---:|---:|---:|
| `llama-3.3-70b-versatile` | 0.98 | 0.67 | 0.53 |

Interpretation:
- `easy` is still quite direct and can be near-perfect for strong LLMs
- `medium` and `hard` are more informative because they require retrieval, escalation judgment, and multi-turn behavior

## Project Structure

```text
helpdesk_env/
├── README.md
├── Dockerfile
├── .gitignore
├── .dockerignore
├── __init__.py
├── client.py
├── data/
│   ├── knowledge_base.json
│   └── tickets/
│       ├── easy.json
│       ├── medium.json
│       └── hard.json
├── inference.py
├── models.py
├── openenv.yaml
├── requirements.txt
├── user_simulator.py
├── graders/
│   ├── category_grader.py
│   ├── faq_grader.py
│   └── resolution_grader.py
└── server/
    ├── app.py
    └── helpdesk_environment.py
```

## Notes

[user_simulator.py](user_simulator.py) is intentionally kept. It powers the customer-side replies for the `hard` task, which is what makes the benchmark genuinely multi-turn instead of a static single-response scoring setup.
