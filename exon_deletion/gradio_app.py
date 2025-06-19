# gradio_app.py
import gradio as gr
import os
from main import run_agent

def process_prompt(user_prompt, wt_file=None, delta_file=None):
    # Handle file uploads and insert paths into the prompt
    if wt_file:
        wt_path = wt_file.name
        user_prompt = user_prompt.replace("wt_file", wt_path)
    if delta_file:
        delta_path = delta_file.name
        user_prompt = user_prompt.replace("delta_file", delta_path)

    # Run the agent with updated prompt
    try:
        response, paths = run_agent(user_prompt)
    except Exception as e:
        response, paths = f"Error: {str(e)}", []

    downloads = []
    for fpath in paths:
        if os.path.exists(fpath):
            downloads.append(fpath)
    
    image_files = []
    for img_path in paths:
        if os.path.exists(img_path) and img_path.endswith('.png'):
            image_files.append(img_path)  # gr.Image accepts file path directly

    return response, downloads, image_files

    # return result or "No output returned."

with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple", secondary_hue="cyan")) as demo:
    gr.Markdown("""
    <h1 style='text-align: center; color: #6A0DAD;'>🧬 Exon Deletion Analyzer</h1>
    <p style='text-align: center; color: #444;'>An End to End Exon Impact Tool</p>
    """)

    with gr.Row():
        with gr.Column(scale=3):  # Make prompt 3x wider
            user_prompt = gr.Textbox(
                placeholder="e.g., Compare regulatory changes using [WT_FILE] and [DELTA_FILE]",
                label="Input Prompt",
                lines=6
            )

            process_button = gr.Button("🧪 Analyze", size="lg")

        with gr.Column(scale=0.5):  # File upload column
            wt_file = gr.File(
                label="Wild-Type FASTA",
                file_types=[".fasta", ".fa"],
                type="filepath",
                height=50  # Makes it visibly smaller
            )
            delta_file = gr.File(
                label="Exon-Deleted FASTA",
                file_types=[".fasta", ".fa"],
                type="filepath",
                height=50
            )

    # with gr.Row():
        

    with gr.Row():
        output_text = gr.Textbox(label="🔬 Output", lines=10, interactive=False, show_copy_button=True)

    with gr.Row():
        download_gallery = gr.Files(label="📁 Download Results")
    
    with gr.Row():
        image_gallery = gr.Gallery(label="📊 Visualizations", columns=2, height="auto")

    def process_ui(prompt, wt, delta):
        response_text, downloads, image_paths = process_prompt(prompt, wt, delta)
        return response_text, downloads, image_paths

    process_button.click(
        fn=process_ui,
        inputs=[user_prompt, wt_file, delta_file],
        outputs=[output_text, download_gallery, image_gallery]
    )

if __name__ == "__main__":
    demo.launch(server_port=7861, share=True)
