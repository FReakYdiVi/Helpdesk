from typing import Iterable


def grade_operation_choice(selected_operation: str, valid_operations: Iterable[str]) -> float:
    operation = selected_operation.strip().lower()
    valid = {candidate.strip().lower() for candidate in valid_operations if candidate.strip()}
    if not operation or not valid:
        return 0.0
    return 1.0 if operation in valid else 0.0


def grade_retrieval_or_action_match(selected_reference: str, gold_reference: str) -> float:
    if selected_reference.strip() and selected_reference.strip() == gold_reference.strip():
        return 1.0
    return 0.0


def grade_escalation(agent_escalated: bool, should_escalate: bool, correct_target: bool = True) -> float:
    if agent_escalated != should_escalate:
        return 0.0
    if agent_escalated and not correct_target:
        return 0.5
    return 1.0


# Backward-compatible alias from the old FAQ-focused environment.
def grade_faq_retrieval(retrieved_faq_id: str, gold_faq_id: str) -> float:
    return grade_retrieval_or_action_match(retrieved_faq_id, gold_faq_id)
