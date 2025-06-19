from graph import create_graph
import warnings
warnings.filterwarnings('ignore')  
if __name__ == "__main__":
    app = create_graph()
    img=app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(img)

    initial_state = {"messages": []}

    try:
        while True:
            # The invoke method now correctly uses the compiled graph
            final_state = app.invoke(initial_state)
            # Reset for the next loop
            initial_state = {"messages": [], "next": None}
    except KeyboardInterrupt:
        print("\nSession ended.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")

# gradio_app.py

# import gradio as gr
# from graph import create_graph

# # Global graph instance
# graph = create_graph()
# graph_executor = graph.get_graph()

# # Global state holder
# state_holder = {
#     "state": {"messages": []},
# }

# # Step 1: User submits compound name
# def submit_compound(compound_name):
#     state = state_holder["state"]
#     state["target_mol"] = compound_name
#     result = graph_executor.invoke(state)
#     state_holder["state"] = result
#     status = f"✅ Compound info fetched for: {compound_name}"
#     reactions = result.get("predicted_reactions", [])
#     return status, gr.update(choices=reactions), "Select reactions for enzyme prediction:"

# # Step 2: User selects reactions from list
# def select_reactions(selected):
#     state = state_holder["state"]
#     state["selected_reactions"] = selected
#     result = graph_executor.invoke(state)
#     state_holder["state"] = result
#     return "✅ Reactions selected. Proceeding with EC number prediction..."

# # Step 3: View final summary
# def generate_summary():
#     result = graph_executor.invoke(state_holder["state"])
#     summary = result.get("summary", "No summary generated.")
#     return summary
