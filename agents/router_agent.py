# agents/router_agent.py

from agents.chat_agent import ChatAgent
from agents.email_agent import EmailAgent
from agents.ivr_agent import IVRAgent
from agents.knowledge_agent import KnowledgeAgent  # 👈 NEW

class RouterAgent:
    def __init__(self):
        self.chat_agent = ChatAgent()
        self.email_agent = EmailAgent()
        self.ivr_agent = IVRAgent()
        self.knowledge_agent = KnowledgeAgent()  # 👈 NEW

    def route(self, query: str, channel: str = "chat"):
        query_lower = query.lower()

        # Channel-based hard routing
        if channel == "email":
            print("Routing to: EmailAgent (Email channel)")
            return self.email_agent.run(query, channel=channel)

        if channel == "ivr":
            print("Routing to: IVRAgent (IVR channel)")
            return self.ivr_agent.run(query, channel=channel)

        # Content-based routing (for chat only)
        if any(keyword in query_lower for keyword in ["policy", "track", "shipping", "faq", "how do i"]):
            print("Routing to: KnowledgeAgent (FAQ-based)")
            return self.knowledge_agent.run(query, channel=channel)

        if "escalate" in query_lower or "complaint" in query_lower:
            print("Routing to: EmailAgent (Escalation via chat)")
            return self.email_agent.run(query, channel=channel)

        if any(keyword in query_lower for keyword in ["refund", "cancel", "return"]):
            print("Routing to: ChatAgent (Returns/Refunds via chat)")
            return self.chat_agent.run(query, channel=channel)

        print("Routing to: ChatAgent (General via chat)")
        return self.chat_agent.run(query, channel=channel)


