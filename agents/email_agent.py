# agents/email_agent.py

import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.callbacks.manager import get_openai_callback
from agents.logger import log_interaction


class EmailAgent:
    def __init__(self, temperature=0.5, model="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(
            temperature=temperature,
            model=model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def run(self, user_input, channel = "email"):
        system_prompt = (
            "You are a professional email assistant for a customer support team. "
            "Write clear, polite, and well-structured email replies to customer inquiries. "
            "Avoid casual language. Format with greetings and sign-offs."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]

        with get_openai_callback() as cb:
            start = time.time()
            response = self.llm.invoke(messages)
            end = time.time()

            print(f"Response: {response.content}")
            print(f"Tokens used: {cb.total_tokens}")
            print(f"Cost: ${cb.total_cost:.6f}")
            print(f"Latency: {end - start:.2f} seconds")

        log_interaction({
            "agent": "EmailAgent",
            "channel": channel,
            "query": user_input,
            "response": response.content,
            "tokens": cb.total_tokens,
            "cost": cb.total_cost,
            "latency": round(end - start, 2)
        })

        return response.content
