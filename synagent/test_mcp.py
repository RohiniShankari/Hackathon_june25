import requests
import sseclient
import time
import json

# Step 1: Start the SSE connection
SSE_URL = "https://agents-mcp-hackathon-mcp-chai1-modal.hf.space/gradio_api/mcp/sse"

print("Connecting to SSE stream...")
sse_response = requests.get(SSE_URL, stream=True)
sse_client = sseclient.SSEClient(sse_response)

session_url = None

# Step 2: Extract session-specific URL
for event in sse_client.events():
    if "/gradio_api/mcp/messages" in event.data:
        session_url = "https://agents-mcp-hackathon-mcp-chai1-modal.hf.space" + event.data
        print(f"Session URL: {session_url}")
        break

# Step 3: Submit a tool call
tool_url = "https://agents-mcp-hackathon-mcp-chai1-modal.hf.space/gradio_api/mcp/tool_call/"
payload = {
    "tool_name": "MCP_Chai1_Modal_create_fasta_file",
    "kwargs": {
        "file_content": "MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGQKL",
        "name": "test_seq.fasta"
    }
}

print("Sending tool call...")
tool_response = requests.post(tool_url, json=payload)
print(f"Tool call response: {tool_response.status_code} - {tool_response.text}")

# Step 4: Listen for results from the session URL
if session_url:
    print("Waiting for output...")
    session_response = requests.get(session_url, stream=True)
    session_client = sseclient.SSEClient(session_response)

    for event in session_client.events():
        if event.data == "[DONE]":
            print("Stream completed.")
            break
        try:
            data = json.loads(event.data)
            print("Tool output:", json.dumps(data, indent=2))
        except json.JSONDecodeError:
            print("Non-JSON event:", event.data)
else:
    print("No session URL received.")
