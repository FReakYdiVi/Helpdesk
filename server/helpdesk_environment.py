import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from graders.category_grader import grade_classification, grade_information_collection
from graders.faq_grader import (
    grade_escalation,
    grade_faq_retrieval,
    grade_operation_choice,
)
from graders.resolution_grader import grade_case_closure, grade_resolution
from models import Action, Observation, Reward, TicketState
from user_simulator import UserSimulator


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


class HelpdeskEnv:
    def __init__(self):
        data_dir = _data_dir()
        tickets_dir = data_dir / "tickets"

        with open(data_dir / "knowledge_base.json", "r", encoding="utf-8") as f:
            self.kb: List[Dict[str, str]] = json.load(f)
        with open(tickets_dir / "easy.json", "r", encoding="utf-8") as f:
            self.easy_tickets: List[Dict[str, Any]] = json.load(f)
        with open(tickets_dir / "medium.json", "r", encoding="utf-8") as f:
            self.medium_tickets: List[Dict[str, Any]] = json.load(f)
        with open(tickets_dir / "hard.json", "r", encoding="utf-8") as f:
            self.hard_tickets: List[Dict[str, Any]] = json.load(f)

        self.current_ticket: Optional[Dict[str, Any]] = None
        self.ticket_state: Optional[TicketState] = None
        self.user_sim: Optional[UserSimulator] = None
        self.task_id: str = "easy"
        self.turn_number: int = 0
        self.conversation_history: List[Dict[str, str]] = []
        self.action_history: List[str] = []

    def reset(self, task_id: str = "easy") -> Observation:
        pool_map = {
            "easy": self.easy_tickets,
            "medium": self.medium_tickets,
            "hard": self.hard_tickets,
        }
        if task_id not in pool_map:
            raise ValueError("task_id must be one of: easy, medium, hard")

        self.task_id = task_id
        self.current_ticket = random.choice(pool_map[task_id])
        self.ticket_state = TicketState(
            ticket_id=self.current_ticket["id"],
            track=self._infer_track(self.current_ticket),
            required_slots=self._required_slots(self.current_ticket, task_id),
        )
        self.user_sim = UserSimulator(self.current_ticket) if task_id == "hard" else None
        self.turn_number = 0
        self.conversation_history = []
        self.action_history = []

        return self.state()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.current_ticket is None or self.ticket_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        current_ticket = self.current_ticket
        ticket_state = self.ticket_state

        canonical_action = self._canonicalize_action(action)
        self.turn_number += 1
        ticket_state.turns_used += 1
        self.action_history.append(canonical_action.action_type)
        self._track_collected_slots(canonical_action)

        action_content = (
            canonical_action.message
            or canonical_action.operation
            or canonical_action.target
            or canonical_action.action_type
        )
        self.conversation_history.append({"role": "agent", "content": action_content})

        done = False
        metrics: Dict[str, float] = {
            "correctness": 0.0,
            "safety": 1.0,
            "resolution": 0.0,
            "efficiency": 0.0,
            "penalties": 0.0,
        }
        info: Dict[str, Any] = {
            "action_type": canonical_action.action_type,
            "operation": canonical_action.operation,
            "target": canonical_action.target,
        }

        if canonical_action.action_type == "ask_for_details":
            metrics["correctness"] = self._grade_detail_request(canonical_action)
            if self.task_id == "hard" and self.user_sim is not None:
                user_response = self.user_sim.respond(canonical_action.message or "")
                self.conversation_history.append({"role": "user", "content": user_response})
                ticket_state.clarification_received = self.user_sim.clarification_given
                info["user_response"] = user_response

        elif canonical_action.action_type == "take_action":
            correctness, resolved = self._grade_take_action(canonical_action)
            metrics["correctness"] = correctness
            ticket_state.issue_resolved = resolved
            if resolved:
                metrics["resolution"] = grade_resolution(ticket_state)
                done = True

        elif canonical_action.action_type == "respond_to_user":
            metrics["correctness"] = self._grade_response(canonical_action)
            if self.task_id == "hard" and self.user_sim is not None:
                user_response = self.user_sim.respond(canonical_action.message or "")
                self.conversation_history.append({"role": "user", "content": user_response})
                ticket_state.issue_resolved = self.user_sim.confirm_resolved()
                info["user_response"] = user_response

        elif canonical_action.action_type == "escalate_case":
            metrics["correctness"] = grade_escalation(
                True,
                bool(current_ticket.get("should_escalate", False)),
            )
            ticket_state.escalated = True
            metrics["resolution"] = metrics["correctness"]
            info["escalation_accuracy"] = metrics["correctness"]
            done = True

        elif canonical_action.action_type == "close_case":
            if self.task_id == "hard" and self.user_sim is not None:
                ticket_state.issue_resolved = self.user_sim.confirm_resolved()
            metrics["resolution"] = grade_case_closure(ticket_state)
            if metrics["resolution"] == 0.0 and not ticket_state.escalated:
                metrics["penalties"] -= 0.20
            done = True

        metrics["safety"] = self._grade_safety(canonical_action, metrics)
        metrics["efficiency"] = self._grade_efficiency(done)

        reward = self._calculate_reward(metrics, done=done)
        info.update(
            {
                "ticket_id": ticket_state.ticket_id,
                "task_id": self.task_id,
                "track": ticket_state.track,
                "turn_number": self.turn_number,
            }
        )
        return self.state(), reward, done, info

    def _canonicalize_action(self, action: Action) -> Action:
        if action.action_type in {
            "ask_for_details",
            "take_action",
            "respond_to_user",
            "escalate_case",
            "close_case",
        }:
            return action

        if action.action_type == "classify":
            return Action(
                action_type="take_action",
                operation="classify_issue",
                category=action.category,
                message=action.message,
            )

        if action.action_type == "lookup_faq":
            return Action(
                action_type="take_action",
                operation="lookup_faq",
                faq_id=action.faq_id,
                message=action.message,
            )

        if action.action_type == "ask_clarification":
            return Action(
                action_type="ask_for_details",
                fields_requested=["issue_details"],
                message=action.message,
            )

        if action.action_type == "reply":
            return Action(
                action_type="respond_to_user",
                message=action.message,
            )

        if action.action_type == "escalate":
            return Action(
                action_type="escalate_case",
                target="human_agent",
                message=action.message,
            )

        if action.action_type == "resolve_ticket":
            return Action(
                action_type="close_case",
                operation="resolve_with_guidance",
                message=action.message,
            )

        raise ValueError(f"Unsupported action type: {action.action_type}")

    def _infer_track(self, ticket: Dict[str, Any]) -> str:
        category = (
            ticket.get("issue_category")
            or ticket.get("gold_category")
            or ticket.get("difficulty")
            or self.task_id
        )
        return str(category).strip().lower().replace(" ", "_")

    def _required_slots(self, ticket: Dict[str, Any], task_id: str) -> List[str]:
        if task_id == "easy":
            return ["issue_category"]
        if task_id == "medium":
            return ["faq_or_escalation_decision"]
        return ["issue_details", "resolution_confirmation"]

    def _track_collected_slots(self, action: Action) -> None:
        if self.ticket_state is None:
            return

        for field_name in action.fields_requested:
            self.ticket_state.collected_slots[field_name] = "requested"

        if action.operation:
            self.ticket_state.collected_slots["last_operation"] = action.operation
        if action.target:
            self.ticket_state.collected_slots["escalation_target"] = action.target

    def _grade_detail_request(self, action: Action) -> float:
        if self.ticket_state is None:
            return 0.0
        if not action.fields_requested and not action.message:
            return 0.0
        if not self.ticket_state.required_slots:
            return 0.5
        info_score = grade_information_collection(
            action.fields_requested,
            self.ticket_state.required_slots,
        )
        if self.task_id != "hard" and info_score == 0.0:
            return 0.5
        return info_score

    def _grade_take_action(self, action: Action) -> Tuple[float, bool]:
        if self.current_ticket is None:
            return 0.0, False

        operation = (action.operation or "").strip().lower()
        current_ticket = self.current_ticket

        if operation == "classify_issue":
            gold_category = current_ticket.get("gold_category", "")
            score = grade_classification(action.category or "", gold_category)
            return score, score == 1.0

        if operation == "lookup_faq":
            gold_faq_id = current_ticket.get("gold_faq_id", "")
            score = grade_faq_retrieval(action.faq_id or "", gold_faq_id)
            if self.ticket_state is not None and score == 1.0:
                self.ticket_state.correct_faq_retrieved = True
            return score, False

        if operation == "resolve_with_guidance":
            resolved = bool(
                self.ticket_state
                and self.ticket_state.correct_faq_retrieved
                and (self.task_id != "hard" or self.ticket_state.clarification_received)
            )
            return (1.0 if resolved else 0.0), resolved

        if operation == "check_status":
            return 0.5, False

        banking_operations = {
            "check_payment",
            "check_refund",
            "check_kyc",
            "secure_account",
            "troubleshoot_upi",
        }
        op_score = grade_operation_choice(operation, banking_operations)
        return op_score, False

    def _grade_response(self, action: Action) -> float:
        if not action.message:
            return 0.0
        if self.task_id == "hard" and self.ticket_state and self.ticket_state.correct_faq_retrieved:
            return 1.0
        return 0.5

    def _grade_safety(self, action: Action, metrics: Dict[str, float]) -> float:
        text = (action.message or "").lower()
        sensitive_markers = ["otp", "pin", "cvv", "password"]
        if any(marker in text for marker in sensitive_markers):
            metrics["penalties"] -= 0.50
            return 0.0

        if action.action_type == "close_case" and metrics["resolution"] == 0.0:
            return 0.25

        if action.action_type == "escalate_case":
            expected = bool(self.current_ticket and self.current_ticket.get("should_escalate", False))
            return 1.0 if expected else 0.6

        return 1.0

    def _grade_efficiency(self, done: bool) -> float:
        max_turns = 1 if self.task_id == "easy" else 2 if self.task_id == "medium" else 6
        if not done:
            remaining_ratio = max(0.0, 1.0 - (self.turn_number / max_turns))
            return round(0.5 * remaining_ratio, 3)
        return max(0.0, min(1.0, 1.0 - (0.1 * max(0, self.turn_number - 1))))

    def _calculate_reward(self, metrics: Dict[str, float], done: bool) -> Reward:
        correctness = metrics.get("correctness", 0.0)
        safety = metrics.get("safety", 0.0)
        resolution = metrics.get("resolution", 0.0)
        efficiency = metrics.get("efficiency", 0.0)
        penalties = metrics.get("penalties", 0.0)

        weighted = (
            (0.35 * correctness)
            + (0.30 * safety)
            + (0.20 * resolution)
            + (0.15 * efficiency)
        )

        recent_actions = self.action_history[-3:]
        if len(recent_actions) >= 2 and len(set(recent_actions)) < len(recent_actions):
            penalties -= 0.05

        case_adjustment = self._case_complexity_adjustment()
        final_value = max(0.0, min(1.0, weighted + penalties + case_adjustment))
        return Reward(
            value=final_value,
            correctness=correctness,
            safety=safety,
            resolution=resolution,
            efficiency=efficiency,
            penalties=penalties,
            done=done,
            info={
                "turn_number": self.turn_number,
                "task_id": self.task_id,
                "case_adjustment": case_adjustment,
                "escalation_accuracy": metrics.get("escalation_accuracy", correctness),
            },
        )

    def _case_complexity_adjustment(self) -> float:
        if self.current_ticket is None:
            return 0.0

        ticket_id = str(self.current_ticket.get("id", ""))
        bucket = sum(ord(char) for char in ticket_id) % 4
        return -0.015 * bucket

    def _build_known_facts(self) -> Dict[str, Any]:
        if self.current_ticket is None or self.ticket_state is None:
            return {}

        return {
            "difficulty": self.current_ticket.get("difficulty", self.task_id),
            "knowledge_base": self.kb,
            "available_categories": [
                "payment_failure",
                "refund_delay",
                "fraud_complaint",
                "kyc_account_restriction",
                "upi_pin_or_bank_linking",
            ],
            "clarification_received": self.ticket_state.clarification_received,
            "faq_retrieved": self.ticket_state.correct_faq_retrieved,
            "issue_resolved": self.ticket_state.issue_resolved,
            "collected_slots": self.ticket_state.collected_slots,
        }

    def state(self) -> Observation:
        if self.current_ticket is None or self.ticket_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        customer_message = self.current_ticket.get("text") or self.current_ticket.get(
            "initial_text", ""
        )
        return Observation(
            case_id=self.current_ticket["id"],
            track=self.task_id,
            customer_message=customer_message,
            conversation_history=self.conversation_history,
            known_facts=self._build_known_facts(),
            required_slots=self.ticket_state.required_slots,
            available_actions=[
                "ask_for_details",
                "take_action",
                "respond_to_user",
                "escalate_case",
                "close_case",
            ],
            turn_number=self.turn_number,
        )


__all__ = ["HelpdeskEnv"]
