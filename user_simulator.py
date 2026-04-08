import random
from typing import Dict, List, Set


GENERIC_DETAIL_KEYWORDS = {
    "utr",
    "amount",
    "time",
    "merchant",
    "bank",
    "transaction",
    "issue",
    "status",
    "error",
}

UNSAFE_MARKERS = {"otp", "pin", "cvv", "password"}

CATEGORY_PROMPTS = {
    "payment_failure": {
        "vague": [
            "The money left my account but the payment does not look right.",
            "Something went wrong with the payment and I need help.",
        ],
        "follow_up": [
            "Can you tell me what I should check next?",
            "What should I do from the app now?",
        ],
        "success": [
            "I checked again and it looks fixed now.",
            "That helped and the payment status is clear now.",
        ],
        "stuck": [
            "I still do not see a proper final status.",
            "It is still not fully clear on my side.",
        ],
    },
    "refund_delay": {
        "vague": [
            "I am still waiting for the money to come back.",
            "The refund is taking too long and I am worried.",
        ],
        "follow_up": [
            "Can you tell me how long this usually takes?",
            "What should I check before I wait more?",
        ],
        "success": [
            "I checked again and the refund is now visible.",
            "Looks like the refund has finally come through.",
        ],
        "stuck": [
            "I still do not see the refund in my bank account.",
            "The refund is still missing for me.",
        ],
    },
    "fraud_complaint": {
        "vague": [
            "This looks suspicious and I need urgent help.",
            "I think something unauthorized happened on my account.",
        ],
        "follow_up": [
            "Please tell me what I should secure right now.",
            "What should I do immediately from my side?",
        ],
        "success": [
            "I have secured the account and followed the steps.",
            "I did the security steps and I understand the next action now.",
        ],
        "stuck": [
            "I am still worried because I do not know if the account is safe yet.",
            "I followed some steps but I still need help with the fraud case.",
        ],
    },
    "kyc_account_restriction": {
        "vague": [
            "The app keeps saying my account is restricted.",
            "I am blocked and not sure what KYC step is missing.",
        ],
        "follow_up": [
            "Can you explain what verification I should complete?",
            "What is the next KYC step I should take?",
        ],
        "success": [
            "That makes sense and I know what verification to do now.",
            "I understand the KYC steps now, thanks.",
        ],
        "stuck": [
            "I still do not understand why it is restricted.",
            "It still looks blocked from my side.",
        ],
    },
    "upi_pin_or_bank_linking": {
        "vague": [
            "The setup keeps failing and I do not know why.",
            "I cannot complete the bank or PIN setup.",
        ],
        "follow_up": [
            "Can you tell me what I should check on the phone first?",
            "What should I try before I do it again?",
        ],
        "success": [
            "That helped and I was able to move forward.",
            "I tried those checks and it is working better now.",
        ],
        "stuck": [
            "It is still failing after I checked that.",
            "I still cannot complete the setup.",
        ],
    },
}


class UserSimulator:
    def __init__(self, ticket: Dict):
        self.ticket_id = ticket.get("id", "")
        self.initial_text = ticket.get("initial_text", "")
        self.clarified_text = ticket.get("clarified_text", "")
        self.trigger_phrases: List[str] = ticket.get("trigger_phrases", [])
        self.gold_faq_id = ticket.get("gold_faq_id", "")
        self.issue_category = ticket.get("issue_category", "")

        self.state = "initial"
        self.issue_resolved = False
        self.clarification_given = False
        self.turns = 0
        self.repeat_questions = 0
        self.guidance_attempts = 0
        self.requested_details: Set[str] = set()

    def _category_messages(self, kind: str) -> List[str]:
        defaults = CATEGORY_PROMPTS.get(self.issue_category) or CATEGORY_PROMPTS[
            "payment_failure"
        ]
        return defaults[kind]

    def _contains_unsafe_request(self, agent_message: str) -> bool:
        lowered = agent_message.lower()
        return any(marker in lowered for marker in UNSAFE_MARKERS)

    def _extract_requested_details(self, agent_message: str) -> Set[str]:
        lowered = agent_message.lower()
        details = {phrase for phrase in self.trigger_phrases if phrase.lower() in lowered}
        details.update(keyword for keyword in GENERIC_DETAIL_KEYWORDS if keyword in lowered)
        return details

    def _asked_for_useful_details(self, agent_message: str) -> bool:
        details = self._extract_requested_details(agent_message)
        new_details = details - self.requested_details
        if details and not new_details:
            self.repeat_questions += 1
        self.requested_details.update(details)
        return bool(details)

    def _looks_like_guidance(self, agent_message: str) -> bool:
        lowered = agent_message.lower()
        generic_guidance = {
            "check",
            "retry",
            "wait",
            "follow",
            "steps",
            "secure",
            "verify",
            "complete",
            "update",
            "confirm",
            "review",
            "reversal",
            "refund",
        }
        category_hints = {
            "payment_failure": {"status", "merchant", "reversal"},
            "refund_delay": {"refund", "merchant", "credited", "bank"},
            "fraud_complaint": {"secure", "fraud", "recent activity", "report"},
            "kyc_account_restriction": {"kyc", "verification", "documents", "review"},
            "upi_pin_or_bank_linking": {"sim", "bank", "device", "permission"},
        }
        hints = category_hints.get(self.issue_category, set())
        return any(token in lowered for token in generic_guidance | hints)

    def respond(self, agent_message: str) -> str:
        self.turns += 1
        agent_message_lower = agent_message.lower()

        if self._contains_unsafe_request(agent_message):
            self.state = "concerned"
            return "I am not comfortable sharing that. Please tell me a safe way to proceed."

        if self.state == "initial":
            if self._asked_for_useful_details(agent_message):
                self.state = "clarified"
                self.clarification_given = True
                return self.clarified_text or random.choice(self._category_messages("vague"))

            if self.turns > 2:
                return "I already explained the issue. Please ask for the details you need."
            return random.choice(
                [
                    "Can you help me understand what details you need?",
                    random.choice(self._category_messages("vague")),
                ]
            )

        if self.state in {"clarified", "concerned"}:
            if self._asked_for_useful_details(agent_message) and self.repeat_questions > 0:
                return "I already shared that part. Can you tell me the next step?"

            if self._looks_like_guidance(agent_message):
                self.guidance_attempts += 1
                self.state = "guided"
                return random.choice(
                    [
                        "Okay, I will try that now.",
                        random.choice(self._category_messages("follow_up")),
                    ]
                )

            if self.turns > 4:
                return "I need a clear next step, not the same questions again."
            return random.choice(self._category_messages("follow_up"))

        if self.state == "guided":
            self.guidance_attempts += 1
            if self.issue_category == "fraud_complaint":
                self.issue_resolved = self.guidance_attempts >= 1
            else:
                self.issue_resolved = self.guidance_attempts >= 2

            if self.issue_resolved:
                self.state = "resolved"
                return random.choice(self._category_messages("success"))

            return random.choice(self._category_messages("stuck"))

        if self.state == "resolved":
            return "Yes, this looks resolved now."

        return random.choice(self._category_messages("follow_up"))

    def confirm_resolved(self) -> bool:
        return self.issue_resolved
