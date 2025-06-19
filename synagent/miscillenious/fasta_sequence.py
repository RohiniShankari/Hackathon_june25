# import requests

# url = "https://agents-mcp-hackathon-mcp-chai1-modal.hf.space/run/create_fasta_file"

# payload = {
#     "data": [">protein1\nMEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP", "protein1.fasta"]
# }

# response = requests.post(url, json=payload)
# print(response.json())
from gradio_client import Client

client = Client("https://agents-mcp-hackathon-mcp-chai1-modal.hf.space")

# Test API view
print(client.view_api())
