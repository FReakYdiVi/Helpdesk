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
- 10 banking FAQ entries in [knowledge_base.json](/Users/shivanshmundra/Downloads/MetaHack/helpdesk-env/envs/helpdesk_env/data/knowledge_base.json)
- 10 `easy` tickets in [easy.json](/Users/shivanshmundra/Downloads/MetaHack/helpdesk-env/envs/helpdesk_env/data/tickets/easy.json)
- 10 `medium` tickets in [medium.json](/Users/shivanshmundra/Downloads/MetaHack/helpdesk-env/envs/helpdesk_env/data/tickets/medium.json)
- 10 `hard` tickets in [hard.json](/Users/shivanshmundra/Downloads/MetaHack/helpdesk-env/envs/helpdesk_env/data/tickets/hard.json)

## Action Space

The public baseline and server currently accept the legacy action names below, which are internally mapped to the compact action model in [models.py](/Users/shivanshmundra/Downloads/MetaHack/helpdesk-env/envs/helpdesk_env/models.py).

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

The model receives an `Observation` object from [models.py](/Users/shivanshmundra/Downloads/MetaHack/helpdesk-env/envs/helpdesk_env/models.py).

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

Rewards are normalized to the range `0.0` to `1.0` in [environment.py](/Users/shivanshmundra/Downloads/MetaHack/helpdesk-env/envs/helpdesk_env/environment.py).

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
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

### Run Tests

```bash
cd /path/to/helpdesk_env
.venv/bin/python -m py_compile environment.py inference.py models.py
```

### Run the Server

```bash
cd /path/to
PYTHONPATH=. /path/to/helpdesk_env/.venv/bin/uvicorn helpdesk_env.server.app:app --host 127.0.0.1 --port 8000
```

### Build the Docker Image

```bash
cd /path/to/helpdesk_env
docker build -t helpdesk-openenv .
docker run --rm -p 8000:8000 helpdesk-openenv
```

### Use the Python Client

```python
from helpdesk_env.client import HelpdeskEnvClient

client = HelpdeskEnvClient("http://127.0.0.1:8000")
result = client.reset("easy")
print(result.observation.customer_message)
```

### Run Inference

```bash
cd /path/to/helpdesk_env
export GROQ_API_KEY=your_key
.venv/bin/python inference.py
```

Optional model override:

```bash
export LLM_MODEL=llama-3.1-8b-instant
export TASK_NAME=medium
```

## Baseline Scores

Latest observed Groq baseline run after removing answer leakage from the observation:

| Model | Easy | Medium | Hard | Average |
|---|---:|---:|---:|---:|
| `llama-3.3-70b-versatile` | 1.00 | 0.60 | 0.59 | 0.73 |

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
├── environment.py
├── inference.py
├── models.py
├── openenv.yaml
├── requirements.txt
├── graders/
│   ├── category_grader.py
│   ├── faq_grader.py
│   └── resolution_grader.py
└── server/
    ├── app.py
    └── helpdesk_environment.py
```
