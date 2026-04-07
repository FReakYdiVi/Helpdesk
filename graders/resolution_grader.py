from ..models import TicketState
from .score_utils import ensure_open_unit_interval


def grade_resolution(ticket_state: TicketState, max_turns: int = 6) -> float:
    if ticket_state.escalated:
        return ensure_open_unit_interval(1.0)

    if not ticket_state.issue_resolved:
        return ensure_open_unit_interval(0.0)

    if ticket_state.turns_used > max_turns:
        return ensure_open_unit_interval(0.0)

    slot_bonus = 0.1 if ticket_state.required_slots and ticket_state.collected_slots else 0.0
    penalty_turns = max(0, ticket_state.turns_used - 3)
    score = 0.9 + slot_bonus - (0.05 * penalty_turns)
    return ensure_open_unit_interval(score)


def grade_case_closure(ticket_state: TicketState) -> float:
    if ticket_state.issue_resolved or ticket_state.escalated:
        return ensure_open_unit_interval(1.0)
    return ensure_open_unit_interval(0.0)


def grade_clarification(asked_clarification: bool, ticket_needed_clarification: bool) -> float:
    if asked_clarification == ticket_needed_clarification:
        return ensure_open_unit_interval(0.25)
    return ensure_open_unit_interval(0.0)
