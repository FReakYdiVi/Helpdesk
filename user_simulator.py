import random
from typing import Dict, List


class UserSimulator:
    def __init__(self, ticket: Dict):
        self.ticket_id = ticket.get("id", "")
        self.initial_text = ticket.get("initial_text", "")
        self.clarified_text = ticket.get("clarified_text", "")
        self.trigger_phrases: List[str] = ticket.get("trigger_phrases", [])
        self.gold_faq_id = ticket.get("gold_faq_id", "")

        self.state = "initial"
        self.issue_resolved = False
        self.clarification_given = False

    def respond(self, agent_message: str) -> str:
        agent_message_lower = agent_message.lower()

        if self.state == "initial":
            if any(phrase.lower() in agent_message_lower for phrase in self.trigger_phrases):
                self.state = "clarified"
                self.clarification_given = True
                return self.clarified_text
            return random.choice(
                [
                    "I'm not sure what you mean",
                    "Can you help me?",
                    "It just stopped working",
                ]
            )

        if self.state == "clarified":
            guidance_keywords = ["try", "follow", "steps", "should", "please"]
            if any(keyword in agent_message_lower for keyword in guidance_keywords):
                self.state = "waiting_resolve"
            return "Ok I will try that, thanks"

        if self.state == "waiting_resolve":
            self.issue_resolved = True
            return "Yes that fixed it!"

        return "Can you help me?"

    def confirm_resolved(self) -> bool:
        return self.issue_resolved
