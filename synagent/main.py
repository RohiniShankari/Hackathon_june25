from graph import create_graph
# import warnings
# warnings.filterwarnings('ignore')  
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

