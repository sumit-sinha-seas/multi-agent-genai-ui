# main.py

from agents.router_agent import RouterAgent

if __name__ == "__main__":
    router = RouterAgent()
    print("Welcome to Unified AI Contact Center.")
    print("Available channels: chat, email, ivr")
    print("Type 'exit' to quit.\n")

    while True:
        channel = input("Choose a channel (chat/email/ivr): ").strip().lower()
        if channel == "exit":
            break
        if channel not in ["chat", "email", "ivr"]:
            print("Invalid channel. Try again.")
            continue

        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response = router.route(user_input, channel)
        print(f"AI ({channel}): {response}\n")


