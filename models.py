from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Observation(BaseModel):
    case_id: str
    track: str
    customer_message: str
    conversation_history: List[Dict[str, str]]
    known_facts: Dict[str, Any]
    required_slots: List[str]
    available_actions: List[str]
    turn_number: int

    @property
    def ticket_id(self) -> str:
        return self.case_id

    @property
    def task_id(self) -> str:
        return str(self.known_facts.get("difficulty", ""))

    @property
    def ticket_text(self) -> str:
        return self.customer_message

    @property
    def knowledge_base(self) -> List[Dict[str, Any]]:
        kb = self.known_facts.get("knowledge_base", [])
        return kb if isinstance(kb, list) else []

    @property
    def available_categories(self) -> List[str]:
        categories = self.known_facts.get("available_categories", [])
        return categories if isinstance(categories, list) else []


class Action(BaseModel):
    action_type: Literal[
        "ask_for_details",
        "take_action",
        "respond_to_user",
        "escalate_case",
        "close_case",
        "classify",
        "lookup_faq",
        "ask_clarification",
        "reply",
        "escalate",
        "resolve_ticket",
    ]
    message: Optional[str] = None
    fields_requested: List[str] = Field(default_factory=list)
    operation: Optional[str] = None
    target: Optional[str] = None

    # Legacy compatibility with the original helpdesk action schema.
    category: Optional[str] = None
    faq_id: Optional[str] = None


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    correctness: float
    safety: float
    resolution: float
    efficiency: float
    penalties: float
    done: bool
    info: Dict[str, Any]

    @property
    def escalation_accuracy(self) -> float:
        return float(self.info.get("escalation_accuracy", self.correctness))


@dataclass
class TicketState:
    ticket_id: str
    track: str
    required_slots: List[str] = field(default_factory=list)
    collected_slots: Dict[str, Any] = field(default_factory=dict)
    issue_resolved: bool = False
    clarification_received: bool = False
    escalated: bool = False
    turns_used: int = 0
    correct_faq_retrieved: bool = False
