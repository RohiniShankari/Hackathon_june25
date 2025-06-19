from gradio_client import Client
from langchain_core.tools import tool
import os
import requests

# Load client
# client = Client("Agents-MCP-Hackathon/MCP_Chai1_Modal")
client = Client("https://agents-mcp-hackathon-mcp-chai1-modal.hf.space")


# Folder setup
BASE_DIR = "/home/boltzmann-labs/synagent"
CIF_FOLDER = os.path.join(BASE_DIR, "cif_files")
PDB_FOLDER = os.path.join(BASE_DIR, "protein_plots")
os.makedirs(CIF_FOLDER, exist_ok=True)
os.makedirs(PDB_FOLDER, exist_ok=True)

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
    """Plots 3D protein structure from result dataframe and stores in protein_plots/ folder."""
    pdb_temp_path = client.predict(result_df, api_name="/plot_protein")
    pdb_filename = os.path.basename(pdb_temp_path)
    local_pdb_path = os.path.join(PDB_FOLDER, pdb_filename)

    try:
        with open(pdb_temp_path, "rb") as src, open(local_pdb_path, "wb") as dst:
            dst.write(src.read())
        print(f"✅ 3D Protein structure saved at: {local_pdb_path}")
        return local_pdb_path
    except Exception as e:
        print(f"❌ Error saving PDB file: {e}")
        return pdb_temp_path  # fallback to temp path

@tool
def show_cif_file(cif_file_url: str) -> str:
    """Downloads and saves CIF file to local directory."""
    try:
        filename = cif_file_url.split("=")[-1]
        local_path = os.path.join(CIF_FOLDER, filename)

        response = requests.get(cif_file_url)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(response.content)

        print(f"✅ CIF file saved to: {local_path}")

        # Optionally preview content
        with open(local_path, "r") as f:
            content = f.read(500)
            print(f"📄 CIF file preview:\n{content}...\n")

        return local_path
    except Exception as e:
        print(f"❌ Failed to download or read CIF file: {e}")
        return f"Failed to download: {cif_file_url}"

# -------- Main flow --------
if __name__ == "__main__":
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

    result = compute_chai1.invoke({
        "fasta_file_name": fasta_result,
        "config_file_name": config_result
    })
    print("Chai1 Result:", result)

    plot_path = plot_protein.invoke({
        "result_df": result
    })
    print("3D Protein Plot saved at:", plot_path)

    top_model_cif = result["data"][0][-1]
    cif_url = f"https://agents-mcp-hackathon-mcp-chai1-modal.hf.space/file={top_model_cif}"
    view_path = show_cif_file.invoke({
        "cif_file_url": cif_url
    })
    print("CIF file saved at:", view_path)
