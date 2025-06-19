from gradio_client import Client

client = Client("Agents-MCP-Hackathon/MCP_Chai1_Modal")
print("Loaded Client ✔")

# # Step 1: Use known existing files
# fasta_file = "chai1_default_input.fasta"
# config_file = "chai1_default_inference.json"

# # Step 2: Call the compute API
# print("Running Chai1 inference with default files...")
# result = client.predict(
#     fasta_file_name=[[fasta_file]],
#     inference_config_file_name=[[config_file]],
#     api_name="/compute_Chai1"
# )
# print("Inference result:")
# print(result)
# Step 1: Create FASTA
fasta_result = client.predict(
    file_content="MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP",
    name="my_test.fasta",
    api_name="/create_fasta_file"
)
print("FASTA file created:", fasta_result)

# Step 2: Create config
config_result = client.predict(
    num_diffn_timesteps=300,
    num_trunk_recycles=3,
    seed=42,
    options=["ESM_embeddings"],
    name="my_config.json",
    api_name="/create_json_config"
)
print("Config file created:", config_result)
result = client.predict(
		api_name="/update_file_explorer"
)
print(result)
