from gradio_client import Client,handle_file
from langchain_core.tools import tool

# Load client
client = Client("Agents-MCP-Hackathon/MCP_Chai1_Modal")

# Define tools using dictionary-based inputs
@tool
def create_fasta_file(file_content: str, filename: str) -> list[list[str]]:
    """Creates a FASTA file from biomolecule sequence."""
    return client.predict(file_content, filename, api_name="/create_fasta_file")

@tool
def create_json_config(
    num_diffn_timesteps: int,
    num_trunk_recycles: int,
    seed: int,
    options: list[str],
    filename: str
) -> list[list[str]]:
    """Creates a JSON config file for simulation."""
    return client.predict(
        num_diffn_timesteps,
        num_trunk_recycles,
        seed,
        options,
        filename,
        api_name="/create_json_config"
    )

@tool
def compute_chai1(fasta_file_name: list[list[str]], config_file_name: list[list[str]]) -> dict:
    """Runs Chai-1 simulation and returns results."""
    return client.predict(fasta_file_name, config_file_name, api_name="/compute_Chai1")

@tool
def plot_protein(result_df: dict) -> str:
    """Plots 3D protein structure from result dataframe."""
    return client.predict(result_df, api_name="/plot_protein")

@tool
def show_cif_file(cif_file_url: str) -> str:
    """Displays the CIF file content from a remote URL."""
    local_file = handle_file(cif_file_url)  # Downloads & wraps file for Gradio
    return client.predict(cif_file=local_file, api_name="/show_cif_file")



# -------- Main flow --------
if __name__ == "__main__":
    # ✅ Use dictionary input for LangChain tool .invoke()
    fasta_result = create_fasta_file.invoke({
        "file_content": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP",
        "filename": "protein1.fasta"
    })
    print("FASTA file:", fasta_result)

    config_result = create_json_config.invoke({
        "num_diffn_timesteps": 300,
        "num_trunk_recycles": 3,
        "seed": 42,
        "options": ["ESM_embeddings"],
        "filename": "config1.json"
    })
    print("Config file:", config_result)

    # ✅ Pass list[list[str]] directly — do NOT double wrap
    result = compute_chai1.invoke({
        "fasta_file_name": fasta_result,
        "config_file_name": config_result
    })
    print("Chai1 Result:", result)


    # 4. Plot Protein (3D)
    plot_path = plot_protein.invoke({
        "result_df": result
    })
    print("3D Protein Plot Path:", plot_path)
    top_model_cif = result["data"][0][-1]  # e.g., '5b29df30-preds.model_idx_4.cif'

    # Full URL that can be handled by `handle_file()`
    cif_url = f"https://agents-mcp-hackathon-mcp-chai1-modal.hf.space/file={top_model_cif}"

    view_path = show_cif_file.invoke({
        "cif_file_url": cif_url
    })
    print("CIF file displayed from:", view_path)


