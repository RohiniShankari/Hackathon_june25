import gradio as gr
from typing import Dict, Any

from agent_tools import extract_compound_info, get_bio_synthesis_reactions,run_other_retro_tools,get_atom_mapping,retrieve_ec_numbers,retrieve_enzymes_from_partial_ec,recommend_reaction_conditions

# Dummy initial state
state = {
    "messages": []
}

# Wrap your function calls (use yours from the full script)
def run_pipeline(target_mol: str, biocat_preference: str):
    try:
        state["target_mol"] = target_mol
        state["biocat_preference"] = biocat_preference

        extract_compound_info(state)
        get_bio_synthesis_reactions(state)
        run_other_retro_tools(state)
        get_atom_mapping(state)

        reactions = state["predicted_reactions"]
        return f"Step-wise reactions:\n\n" + "\n".join(reactions)
    
    except Exception as e:
        return str(e)

def select_reactions_gui(reaction_indices: str):
    try:
        indices = [int(i.strip()) for i in reaction_indices.split(",") if i.strip().isdigit()]
        state["selected_reactions"] = [state["predicted_reactions"][i] for i in indices]
        retrieve_ec_numbers(state)
        retrieve_enzymes_from_partial_ec(state)
        recommend_reaction_conditions(state)
        return f"Selected: {state['selected_reactions']}\n\nPredicted ECs: {state['predicted_ec_numbers']}"
    except Exception as e:
        return str(e)

# UI layout
with gr.Blocks() as demo:
    gr.Markdown("## 🧬 Retrosynthesis Gradio Agent")

    with gr.Row():
        target_input = gr.Textbox(label="Enter Target Molecule Name")
        biocat_pref = gr.Textbox(label="Enter Biocatalysis Preference")

    run_btn = gr.Button("Run Pipeline")
    output_display = gr.Textbox(label="Pipeline Output", lines=10)

    run_btn.click(run_pipeline, inputs=[target_input, biocat_pref], outputs=output_display)

    gr.Markdown("### ✅ Select Reactions for EC Number Prediction")
    reaction_input = gr.Textbox(label="Enter Indices (e.g. 0,1,3 or all)")
    ec_btn = gr.Button("Predict EC & Enzymes")
    ec_output = gr.Textbox(label="EC and Enzyme Output", lines=8)

    ec_btn.click(select_reactions_gui, inputs=reaction_input, outputs=ec_output)

demo.launch()
