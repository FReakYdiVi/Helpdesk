"""FastAPI server exposing HelpdeskEnv over HTTP."""

import json
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from .helpdesk_environment import HelpdeskEnv
from ..models import Action, Reward, normalize_action

app = FastAPI(title="Helpdesk OpenEnv")
_env: Optional[HelpdeskEnv] = None

CATEGORIES = [
    "payment_failure",
    "refund_delay",
    "fraud_complaint",
    "kyc_account_restriction",
    "upi_pin_or_bank_linking",
]

FAQ_OPTIONS = [f"faq_{idx:03d}" for idx in range(1, 41)]

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


def _ui_template() -> str:
    tasks_json = json.dumps(TASKS)
    categories_json = json.dumps(CATEGORIES)
    faq_json = json.dumps(FAQ_OPTIONS)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UPI Banking Support Environment</title>
  <style>
    :root {{
      --bg: #efe6d4;
      --panel: rgba(250, 247, 241, 0.92);
      --panel-strong: rgba(255, 252, 246, 0.98);
      --stroke: rgba(111, 98, 84, 0.16);
      --text: #2f2b27;
      --muted: #7a6d60;
      --teal: #173f48;
      --teal-soft: #2b6868;
      --green: #299868;
      --orange: #e67d23;
      --blue: #4b78df;
      --violet: #8c6cb8;
      --danger: #c45757;
      --shadow: 0 18px 40px rgba(70, 55, 33, 0.11);
      --radius-xl: 32px;
      --radius-lg: 22px;
      --radius-md: 18px;
      --radius-sm: 14px;
      --mono: "SFMono-Regular", "JetBrains Mono", "Menlo", monospace;
      --sans: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.72), transparent 24%),
        radial-gradient(circle at top right, rgba(123, 169, 255, 0.12), transparent 18%),
        radial-gradient(circle at bottom right, rgba(165, 101, 189, 0.18), transparent 22%),
        linear-gradient(180deg, #f4ede0 0%, #eee4d2 100%);
    }}

    .page {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 24px;
    }}

    .hero {{
      position: relative;
      overflow: hidden;
      background:
        radial-gradient(circle at 85% 85%, rgba(214, 170, 235, 0.42), transparent 18%),
        linear-gradient(135deg, #22535b 0%, #163847 100%);
      color: #fbf4ea;
      border-radius: 36px;
      padding: 28px 28px 30px;
      box-shadow: var(--shadow);
    }}

    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -14% -40% auto;
      width: 460px;
      height: 460px;
      background: radial-gradient(circle, rgba(255,255,255,0.11), transparent 60%);
      pointer-events: none;
    }}

    .eyebrow,
    .section-title,
    .panel-label,
    .card-kicker {{
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: rgba(255, 245, 232, 0.82);
      font-size: 0.86rem;
      font-weight: 700;
    }}

    .hero-grid {{
      display: grid;
      grid-template-columns: 1.45fr 280px;
      gap: 20px;
      align-items: start;
    }}

    .hero h1 {{
      margin: 12px 0 18px;
      font-size: clamp(3rem, 8vw, 5.4rem);
      line-height: 0.95;
      letter-spacing: -0.05em;
    }}

    .hero p {{
      max-width: 920px;
      margin: 0 0 24px;
      color: rgba(250, 243, 236, 0.88);
      font-size: clamp(1.05rem, 2vw, 1.45rem);
      line-height: 1.45;
    }}

    .hero-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
    }}

    .hero-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 46px;
      padding: 0 18px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.15);
      background: rgba(255, 255, 255, 0.09);
      color: #fff9f0;
      text-decoration: none;
      font-size: 0.98rem;
      font-weight: 600;
      backdrop-filter: blur(8px);
    }}

    .env-pill {{
      display: flex;
      flex-direction: column;
      gap: 14px;
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255,255,255,0.13);
      border-radius: 28px;
      padding: 24px 22px;
      backdrop-filter: blur(10px);
      min-height: 140px;
    }}

    .env-status {{
      font-size: 2rem;
      font-weight: 800;
      text-transform: lowercase;
      color: #fff8ef;
    }}

    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 18px;
      margin-top: 24px;
    }}

    .stat-card {{
      border-radius: 28px;
      padding: 24px 28px;
      color: #fff8ef;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }}

    .stat-card.reward {{ background: linear-gradient(135deg, #299868 0%, #25885f 100%); }}
    .stat-card.track {{ background: linear-gradient(135deg, #e67d23 0%, #df9027 100%); }}
    .stat-card.turn {{ background: linear-gradient(135deg, #3f69c5 0%, #4c79df 100%); }}
    .stat-card.status {{ background: linear-gradient(135deg, #6e5493 0%, #9977a4 100%); }}

    .stat-value {{
      font-size: clamp(2.6rem, 5vw, 4rem);
      font-weight: 900;
      line-height: 1;
      margin: 18px 0 12px;
      letter-spacing: -0.04em;
    }}

    .stat-note {{
      font-size: 0.96rem;
      color: rgba(255,248,239,0.84);
    }}

    .content-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(380px, 0.95fr);
      gap: 22px;
      margin-top: 22px;
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.45);
      border-radius: var(--radius-xl);
      padding: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}

    .section-title {{
      color: var(--muted);
      margin-bottom: 22px;
      font-size: 0.9rem;
    }}

    .ticket-box,
    .message-box,
    .mini-card,
    .console-card,
    .timeline-event,
    .status-banner,
    .detail-box {{
      background: var(--panel-strong);
      border: 1px solid var(--stroke);
      border-radius: var(--radius-lg);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.5);
    }}

    .message-box {{
      min-height: 220px;
      padding: 26px;
      margin-bottom: 18px;
    }}

    .message-box p {{
      margin: 14px 0 0;
      font-size: clamp(1.4rem, 2.8vw, 1.8rem);
      line-height: 1.35;
    }}

    .status-banner {{
      padding: 20px 22px;
      background: linear-gradient(180deg, rgba(227, 236, 231, 0.96), rgba(233, 239, 234, 0.96));
      border-color: rgba(61, 151, 117, 0.24);
      color: #2a6360;
      font-size: 1rem;
      line-height: 1.45;
      margin-bottom: 18px;
    }}

    .mini-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}

    .mini-card,
    .detail-box {{
      padding: 20px 22px;
      min-height: 170px;
    }}

    .panel-label,
    .card-kicker {{
      color: var(--muted);
      font-size: 0.82rem;
    }}

    .mini-value,
    .detail-box pre,
    .code-block {{
      margin-top: 12px;
      font-size: 1rem;
      line-height: 1.55;
    }}

    .code-block,
    .detail-box pre {{
      font-family: var(--mono);
      white-space: pre-wrap;
      word-break: break-word;
      margin: 10px 0 0;
      color: #3f372f;
    }}

    .console-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-bottom: 16px;
    }}

    .field {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}

    label {{
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-size: 0.8rem;
      color: var(--muted);
      font-weight: 700;
    }}

    select,
    textarea,
    input {{
      width: 100%;
      border-radius: 18px;
      border: 1px solid rgba(129, 115, 98, 0.22);
      background: rgba(255,255,255,0.92);
      padding: 16px 18px;
      font: inherit;
      color: var(--text);
      outline: none;
      transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
    }}

    select:focus,
    textarea:focus,
    input:focus {{
      border-color: rgba(58, 120, 122, 0.48);
      box-shadow: 0 0 0 4px rgba(45, 128, 120, 0.09);
    }}

    textarea {{
      resize: vertical;
      min-height: 110px;
    }}

    .button-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin: 10px 0 14px;
    }}

    button {{
      border: 0;
      min-height: 54px;
      padding: 0 28px;
      border-radius: 999px;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      transition: transform 140ms ease, opacity 140ms ease, box-shadow 140ms ease;
    }}

    button:hover {{
      transform: translateY(-1px);
    }}

    button:disabled {{
      opacity: 0.65;
      cursor: wait;
      transform: none;
    }}

    .btn-primary {{
      background: linear-gradient(135deg, #27976b 0%, #20895f 100%);
      color: #f8f6f0;
      box-shadow: 0 12px 22px rgba(41, 152, 104, 0.18);
    }}

    .btn-secondary {{
      background: #dfe4ec;
      color: #31507f;
    }}

    .btn-tertiary {{
      background: #ede8df;
      color: #2d2a26;
    }}

    .helper-text {{
      color: var(--muted);
      font-size: 0.98rem;
      line-height: 1.5;
      margin-top: 8px;
    }}

    .timeline-list {{
      display: flex;
      flex-direction: column;
      gap: 14px;
      min-height: 520px;
    }}

    .timeline-event {{
      padding: 18px 20px;
    }}

    .timeline-event.waiting {{
      border-style: dashed;
      color: var(--muted);
    }}

    .timeline-meta {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 0.9rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-weight: 700;
    }}

    .timeline-event pre {{
      margin: 0;
      font-family: var(--mono);
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.5;
    }}

    .details-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}

    .details-grid .detail-box:last-child {{
      grid-column: 1 / -1;
      min-height: 220px;
    }}

    .reward-breakdown-row {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin: 8px 0;
      font-family: var(--mono);
    }}

    .muted {{
      color: var(--muted);
    }}

    .hidden {{
      display: none;
    }}

    @media (max-width: 1180px) {{
      .hero-grid,
      .content-grid {{
        grid-template-columns: 1fr;
      }}

      .stat-grid {{
        grid-template-columns: repeat(2, 1fr);
      }}
    }}

    @media (max-width: 720px) {{
      .page {{
        padding: 14px;
      }}

      .hero,
      .panel {{
        padding: 18px;
        border-radius: 26px;
      }}

      .stat-grid,
      .mini-grid,
      .console-grid,
      .details-grid {{
        grid-template-columns: 1fr;
      }}

      .button-row {{
        flex-direction: column;
      }}

      button {{
        width: 100%;
      }}

      .timeline-list {{
        min-height: auto;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">HF Space Dashboard</div>
          <h1>UPI Banking Support Environment</h1>
          <p>
            Run the benchmark like an operator: reset an episode, choose the exact action
            your agent would take, and inspect the live observation, conversation, and
            current reward after each step.
          </p>
          <div class="hero-links">
            <a class="hero-link" href="/health" target="_blank" rel="noreferrer">Health</a>
            <a class="hero-link" href="/docs" target="_blank" rel="noreferrer">API Docs</a>
            <a class="hero-link" href="/state" target="_blank" rel="noreferrer">Raw State</a>
          </div>
        </div>
        <div class="env-pill">
          <div class="eyebrow">Environment</div>
          <div id="env-status" class="env-status">checking</div>
          <div class="muted">FastAPI + OpenEnv-style benchmark runtime</div>
        </div>
      </div>

      <div class="stat-grid">
        <div class="stat-card reward">
          <div class="card-kicker">Current Reward</div>
          <div id="hero-reward" class="stat-value">0.000</div>
          <div class="stat-note">Most recent reward value</div>
        </div>
        <div class="stat-card track">
          <div class="card-kicker">Difficulty</div>
          <div id="hero-track" class="stat-value">-</div>
          <div class="stat-note">Current episode track</div>
        </div>
        <div class="stat-card turn">
          <div class="card-kicker">Turn</div>
          <div id="hero-turn" class="stat-value">0</div>
          <div class="stat-note">Current step count</div>
        </div>
        <div class="stat-card status">
          <div class="card-kicker">Status</div>
          <div id="hero-episode-status" class="stat-value">Idle</div>
          <div class="stat-note">Episode completion</div>
        </div>
      </div>
    </section>

    <div class="content-grid">
      <section class="panel">
        <div class="section-title">Current Ticket</div>
        <div class="message-box">
          <div class="panel-label">Customer Message</div>
          <p id="customer-message">Reset the environment to load a ticket.</p>
        </div>

        <div id="status-banner" class="status-banner">
          No progress flags are active yet. Choose the next action based on the ticket and available workflow.
        </div>

        <div class="mini-grid">
          <div class="mini-card">
            <div class="panel-label">Case</div>
            <div id="case-id" class="mini-value">-</div>
          </div>
          <div class="mini-card">
            <div class="panel-label">Required Slots</div>
            <div id="required-slots" class="mini-value">None</div>
          </div>
          <div class="mini-card">
            <div class="panel-label">Available Actions</div>
            <div id="available-actions" class="mini-value">None</div>
          </div>
          <div class="mini-card">
            <div class="panel-label">Collected Facts</div>
            <div id="collected-facts" class="mini-value">-</div>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="section-title">Action Console</div>
        <div class="console-grid">
          <div class="field">
            <label for="task-select">Difficulty</label>
            <select id="task-select"></select>
          </div>
          <div class="field">
            <label for="action-select">Action Type</label>
            <select id="action-select"></select>
          </div>
        </div>

        <div class="console-grid">
          <div class="field" id="category-field">
            <label for="category-select">Category</label>
            <select id="category-select"></select>
          </div>
          <div class="field hidden" id="faq-field">
            <label for="faq-select">FAQ ID</label>
            <select id="faq-select"></select>
          </div>
        </div>

        <div class="field" id="message-field">
          <label for="message-input">Message</label>
          <textarea id="message-input" placeholder="Optional agent message or clarification prompt"></textarea>
        </div>

        <div class="button-row">
          <button id="step-btn" class="btn-primary">Execute Step</button>
          <button id="reset-btn" class="btn-secondary">Reset</button>
          <button id="refresh-btn" class="btn-tertiary">Refresh State</button>
        </div>

        <div id="action-hint" class="helper-text">
          Predict the banking issue category for the current ticket.
        </div>
      </section>
    </div>

    <div class="content-grid">
      <section class="panel">
        <div class="section-title">Conversation Timeline</div>
        <div id="timeline" class="timeline-list">
          <div class="timeline-event waiting">
            <div class="panel-label">Waiting</div>
            <div class="mini-value">No actions yet. Reset the env, then execute a step.</div>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="section-title">Step Details</div>
        <div class="details-grid">
          <div class="detail-box">
            <div class="panel-label">Current Reward Breakdown</div>
            <div id="reward-breakdown" class="code-block">No step executed yet.</div>
          </div>
          <div class="detail-box">
            <div class="panel-label">Episode Info</div>
            <pre id="episode-info">{{\n  "done": false\n}}</pre>
          </div>
          <div class="detail-box">
            <div class="panel-label">Observation Snapshot</div>
            <pre id="observation-snapshot">-</pre>
          </div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const TASKS = {tasks_json};
    const CATEGORIES = {categories_json};
    const FAQ_OPTIONS = {faq_json};

    const state = {{
      observation: null,
      reward: null,
      done: false,
      info: {{}},
      timeline: [],
      activeTask: "easy",
    }};

    const taskSelect = document.getElementById("task-select");
    const actionSelect = document.getElementById("action-select");
    const categorySelect = document.getElementById("category-select");
    const faqSelect = document.getElementById("faq-select");
    const messageInput = document.getElementById("message-input");
    const faqField = document.getElementById("faq-field");
    const categoryField = document.getElementById("category-field");
    const actionHint = document.getElementById("action-hint");
    const stepBtn = document.getElementById("step-btn");
    const resetBtn = document.getElementById("reset-btn");
    const refreshBtn = document.getElementById("refresh-btn");

    const actionOptions = [
      {{ value: "classify", label: "classify" }},
      {{ value: "lookup_faq", label: "lookup_faq" }},
      {{ value: "ask_clarification", label: "ask_clarification" }},
      {{ value: "reply", label: "reply" }},
      {{ value: "escalate", label: "escalate" }},
      {{ value: "resolve_ticket", label: "resolve_ticket" }},
    ];

    const actionHints = {{
      classify: "Predict the banking issue category for the current ticket.",
      lookup_faq: "Choose the FAQ that best matches the current issue and conversation context.",
      ask_clarification: "Ask for the missing details needed to continue safely.",
      reply: "Send a safe support response grounded in the available facts and FAQ guidance.",
      escalate: "Escalate when the issue needs manual handling, review, or fraud intervention.",
      resolve_ticket: "Close the case only when the user issue appears properly resolved.",
    }};

    function initForm() {{
      TASKS.forEach((task) => {{
        const option = document.createElement("option");
        option.value = task.id;
        option.textContent = task.difficulty.charAt(0).toUpperCase() + task.difficulty.slice(1);
        taskSelect.appendChild(option);
      }});

      actionOptions.forEach((action) => {{
        const option = document.createElement("option");
        option.value = action.value;
        option.textContent = action.label;
        actionSelect.appendChild(option);
      }});

      const categoryPlaceholder = document.createElement("option");
      categoryPlaceholder.value = "";
      categoryPlaceholder.textContent = "Select category";
      categorySelect.appendChild(categoryPlaceholder);
      CATEGORIES.forEach((category) => {{
        const option = document.createElement("option");
        option.value = category;
        option.textContent = category;
        categorySelect.appendChild(option);
      }});

      const faqPlaceholder = document.createElement("option");
      faqPlaceholder.value = "";
      faqPlaceholder.textContent = "Select FAQ";
      faqSelect.appendChild(faqPlaceholder);
      FAQ_OPTIONS.forEach((faqId) => {{
        const option = document.createElement("option");
        option.value = faqId;
        option.textContent = faqId;
        faqSelect.appendChild(option);
      }});

      taskSelect.value = state.activeTask;
      actionSelect.value = "classify";
      syncActionFields();
    }}

    function setBusy(isBusy) {{
      [stepBtn, resetBtn, refreshBtn].forEach((btn) => {{
        btn.disabled = isBusy;
      }});
    }}

    function formatPretty(value) {{
      return JSON.stringify(value, null, 2);
    }}

    function titleCase(value) {{
      if (!value) return "-";
      return value.charAt(0).toUpperCase() + value.slice(1);
    }}

    function syncActionFields() {{
      const action = actionSelect.value;
      categoryField.classList.toggle("hidden", action !== "classify");
      faqField.classList.toggle("hidden", action !== "lookup_faq");
      messageInput.parentElement.classList.toggle("hidden", action === "classify" || action === "lookup_faq" || action === "resolve_ticket");
      actionHint.textContent = actionHints[action] || "";
    }}

    function updateHero() {{
      const rewardValue = state.reward?.value ?? 0;
      document.getElementById("hero-reward").textContent = Number(rewardValue).toFixed(3);
      document.getElementById("hero-track").textContent = state.activeTask ? titleCase(state.activeTask) : "-";
      document.getElementById("hero-turn").textContent = String(state.observation?.turn_number ?? 0);
      document.getElementById("hero-episode-status").textContent = state.done ? "Done" : (state.observation ? "Active" : "Idle");
    }}

    function updateEnvironmentStatus(statusText, ok = true) {{
      const el = document.getElementById("env-status");
      el.textContent = statusText;
      el.style.color = ok ? "#fff8ef" : "#ffd2d2";
    }}

    function updateTicketPanel() {{
      const obs = state.observation;
      document.getElementById("customer-message").textContent =
        obs?.customer_message || "Reset the environment to load a ticket.";
      document.getElementById("case-id").textContent = obs?.case_id || "-";
      document.getElementById("required-slots").textContent =
        obs?.required_slots?.length ? obs.required_slots.join(", ") : "None";
      document.getElementById("available-actions").textContent =
        obs?.available_actions?.length ? obs.available_actions.join(", ") : "None";

      const knownFacts = obs?.known_facts || {{}};
      const collectedFacts = knownFacts.collected_slots && Object.keys(knownFacts.collected_slots).length
        ? formatPretty(knownFacts.collected_slots)
        : "-";
      document.getElementById("collected-facts").textContent = collectedFacts;

      const flags = [];
      if (knownFacts.clarification_received) flags.push("Clarification captured");
      if (knownFacts.faq_retrieved) flags.push("Correct FAQ retrieved");
      if (knownFacts.issue_resolved) flags.push("Issue marked resolved");
      if (!flags.length) {{
        flags.push("No progress flags are active yet. Choose the next action based on the ticket and available workflow.");
      }}
      document.getElementById("status-banner").textContent = flags.join(" | ");
    }}

    function updateRewardPanel() {{
      const reward = state.reward;
      const breakdown = document.getElementById("reward-breakdown");
      if (!reward) {{
        breakdown.textContent = "No step executed yet.";
        return;
      }}

      const lines = [
        ["current_reward", Number(reward.value || 0).toFixed(3)],
        ["correctness", reward.correctness ?? 0],
        ["safety", reward.safety ?? 0],
        ["resolution", reward.resolution ?? 0],
        ["efficiency", reward.efficiency ?? 0],
        ["penalties", reward.penalties ?? 0],
      ];
      breakdown.innerHTML = lines.map(([key, value]) => (
        `<div class="reward-breakdown-row"><span>${{key}}</span><span>${{value}}</span></div>`
      )).join("");
    }}

    function updateDetailPanel() {{
      document.getElementById("episode-info").textContent = formatPretty({{
        done: state.done,
        ...state.info,
      }});

      const obs = state.observation ? {{
        case_id: state.observation.case_id,
        task: state.activeTask,
        turn_number: state.observation.turn_number,
        required_slots: state.observation.required_slots,
        conversation_history: state.observation.conversation_history,
        known_facts: state.observation.known_facts,
      }} : "-";
      document.getElementById("observation-snapshot").textContent =
        typeof obs === "string" ? obs : formatPretty(obs);
    }}

    function renderTimeline() {{
      const timeline = document.getElementById("timeline");
      if (!state.timeline.length) {{
        timeline.innerHTML = `
          <div class="timeline-event waiting">
            <div class="panel-label">Waiting</div>
            <div class="mini-value">No actions yet. Reset the env, then execute a step.</div>
          </div>
        `;
        return;
      }}

      timeline.innerHTML = state.timeline.map((entry, index) => `
        <div class="timeline-event">
          <div class="timeline-meta">
            <span>${{entry.kind}}</span>
            <span>step ${{index + 1}}</span>
          </div>
          <pre>${{entry.body}}</pre>
        </div>
      `).join("");
    }}

    function refreshView() {{
      updateHero();
      updateTicketPanel();
      updateRewardPanel();
      updateDetailPanel();
      renderTimeline();
    }}

    async function apiRequest(path, options = {{}}) {{
      const response = await fetch(path, {{
        headers: {{
          "Content-Type": "application/json",
          ...(options.headers || {{}}),
        }},
        ...options,
      }});
      if (!response.ok) {{
        const text = await response.text();
        throw new Error(text || `Request failed with status ${{response.status}}`);
      }}
      return response.json();
    }}

    async function checkHealth() {{
      try {{
        const data = await apiRequest("/health");
        updateEnvironmentStatus(data.status || "healthy", true);
      }} catch (_error) {{
        updateEnvironmentStatus("offline", false);
      }}
    }}

    async function handleReset() {{
      setBusy(true);
      try {{
        state.activeTask = taskSelect.value;
        const data = await apiRequest("/reset", {{
          method: "POST",
          body: JSON.stringify({{ task_id: state.activeTask }}),
        }});
        state.observation = data.observation;
        state.reward = data.reward;
        state.done = data.done;
        state.info = data.info || {{}};
        state.timeline = [
          {{
            kind: "Reset",
            body: `Loaded task=${{state.activeTask}} case=${{data.observation.case_id}}\\ncustomer_message=${{data.observation.customer_message}}`,
          }},
        ];
        refreshView();
      }} catch (error) {{
        state.timeline.unshift({{ kind: "Error", body: String(error) }});
        renderTimeline();
      }} finally {{
        setBusy(false);
      }}
    }}

    function buildActionPayload() {{
      const actionType = actionSelect.value;
      const payload = {{ action_type: actionType }};

      if (actionType === "classify" && categorySelect.value) {{
        payload.category = categorySelect.value;
      }}
      if (actionType === "lookup_faq" && faqSelect.value) {{
        payload.faq_id = faqSelect.value;
      }}
      if (messageInput.value.trim() && actionType !== "classify" && actionType !== "lookup_faq" && actionType !== "resolve_ticket") {{
        payload.message = messageInput.value.trim();
      }}
      return payload;
    }}

    async function handleStep() {{
      setBusy(true);
      try {{
        const action = buildActionPayload();
        const data = await apiRequest("/step", {{
          method: "POST",
          body: JSON.stringify({{ action }}),
        }});
        state.observation = data.observation;
        state.reward = data.reward;
        state.done = data.done;
        state.info = data.info || {{}};
        state.timeline.unshift({{
          kind: data.done ? "Completed" : "Action",
          body:
            `action=${{formatPretty(action)}}\\n` +
            `reward=${{Number(data.reward?.value || 0).toFixed(3)}}\\n` +
            `done=${{String(data.done)}}\\n` +
            `info=${{formatPretty(data.info || {{}})}}`,
        }});
        refreshView();
      }} catch (error) {{
        state.timeline.unshift({{ kind: "Error", body: String(error) }});
        renderTimeline();
      }} finally {{
        setBusy(false);
      }}
    }}

    async function handleRefreshState() {{
      setBusy(true);
      try {{
        const data = await apiRequest("/state");
        state.observation = data.observation;
        state.timeline.unshift({{
          kind: "Refresh",
          body: `state refreshed for case=${{data.observation.case_id}}`,
        }});
        refreshView();
      }} catch (error) {{
        state.timeline.unshift({{ kind: "Error", body: String(error) }});
        renderTimeline();
      }} finally {{
        setBusy(false);
      }}
    }}

    actionSelect.addEventListener("change", syncActionFields);
    taskSelect.addEventListener("change", () => {{
      state.activeTask = taskSelect.value;
      updateHero();
    }});
    resetBtn.addEventListener("click", handleReset);
    stepBtn.addEventListener("click", handleStep);
    refreshBtn.addEventListener("click", handleRefreshState);

    initForm();
    refreshView();
    checkHealth();
  </script>
</body>
</html>"""


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(_ui_template())


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
