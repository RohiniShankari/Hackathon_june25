from graph import create_graph

def run_pipeline(compound_name: str) -> str:
    """
    Runs the full LangGraph pipeline and returns a full log of messages.
    """
    app = create_graph()

    # Start with the user message
    initial_state = {
        "messages": [
            {"role": "user", "content": compound_name}
        ]
    }

    try:
        # Run the LangGraph agent
        final_state = app.invoke(initial_state)

        # Extract all messages for logging
        messages = final_state.get("messages", [])
        chat_log = ""
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role and content:
                prefix = "🧑 User:" if role == "user" else "🤖 Assistant:"
                chat_log += f"{prefix} {content}\n\n"

        return chat_log.strip()

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error occurred: {str(e)}"
