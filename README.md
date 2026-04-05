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
- 10 banking FAQ entries in [data/knowledge_base.json](data/knowledge_base.json)
- 10 `easy` tickets in [data/tickets/easy.json](data/tickets/easy.json)
- 10 `medium` tickets in [data/tickets/medium.json](data/tickets/medium.json)
- 10 `hard` tickets in [data/tickets/hard.json](data/tickets/hard.json)

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

Runtime configuration is read from `.env`.
The environment currently uses:
- `API_BASE_URL=https://api.groq.com/openai/v1`
- `MODEL_NAME=llama-3.3-70b-versatile`
- `GROQ_API_KEY` for model access
- `HF_SPACE_URL` for the deployed Space runtime URL
- `HF_SPACE_TOKEN` for protected Space access when required

## Usage

### Using Docker

```bash
# Build the image from the repository root
docker build -t helpdesk-openenv:latest -f server/Dockerfile .

# Run the server
docker run -p 8000:8000 helpdesk-openenv:latest
```

### Local Development

```bash
# Install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Quick compile check
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  inference.py server/app.py server/helpdesk_environment.py

# Run the server locally
PYTHONPATH=.. .venv/bin/uvicorn helpdesk_env.server.app:app --host 127.0.0.1 --port 8000
```

### Run Inference

```bash
cd /path/to/helpdesk_env
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
curl -H "Authorization: Bearer $HF_SPACE_TOKEN" \
  "$HF_SPACE_URL/health"

curl -X POST "$HF_SPACE_URL/reset" \
  -H "Authorization: Bearer $HF_SPACE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy"}'

curl -X POST "$HF_SPACE_URL/step" \
  -H "Authorization: Bearer $HF_SPACE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"action":{"action_type":"classify","category":"payment_failure"}}'
```

Important:
- Use the `.hf.space` runtime URL for API calls
- Do not use `https://huggingface.co/spaces/...` for `/health` or `/reset`, because that is the Space webpage, not the running API
- If your Space is public, the `Authorization` header is optional

## Hugging Face Space Deployment

This repo is configured as a Docker-based HF Space through the YAML frontmatter at the top of this README:
- `sdk: docker`
- `app_port: 8000`
- `tags` include `openenv`

Typical flow:

```bash
git clone https://huggingface.co/spaces/<username>/<space-name>
cd <space-name>
rsync -av --exclude '.git' /path/to/helpdesk_env/ ./
git add .
git commit -m "Deploy UPI banking support environment"
git push
```

## Baseline Scores

Latest observed Groq baseline run after removing answer leakage from the observation:

| Model | Easy | Medium | Hard |
|---|---:|---:|---:|---:|
| `llama-3.3-70b-versatile` | 0.98 | 0.67 | 0.53 |

Interpretation:
- `easy` is still quite direct and can be near-perfect for strong LLMs
- `medium` and `hard` are more informative because they require retrieval, escalation judgment, and multi-turn behavior

## Project Structure

```text
helpdesk_env/
├── README.md
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
    ├── Dockerfile
    └── helpdesk_environment.py
```

## Notes

[user_simulator.py](user_simulator.py) is intentionally kept. It powers the customer-side replies for the `hard` task, which is what makes the benchmark genuinely multi-turn instead of a static single-response scoring setup.
