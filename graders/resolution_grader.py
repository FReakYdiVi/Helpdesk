from models import TicketState


def grade_resolution(ticket_state: TicketState, max_turns: int = 6) -> float:
    if ticket_state.escalated:
        return 1.0

    if not ticket_state.issue_resolved:
        return 0.0

    if ticket_state.turns_used > max_turns:
        return 0.0

    slot_bonus = 0.1 if ticket_state.required_slots and ticket_state.collected_slots else 0.0
    penalty_turns = max(0, ticket_state.turns_used - 3)
    score = 0.9 + slot_bonus - (0.05 * penalty_turns)
    return max(0.0, min(1.0, score))


def grade_case_closure(ticket_state: TicketState) -> float:
    if ticket_state.issue_resolved or ticket_state.escalated:
        return 1.0
    return 0.0


def grade_clarification(asked_clarification: bool, ticket_needed_clarification: bool) -> float:
    if asked_clarification == ticket_needed_clarification:
        return 0.25
    return 0.0
