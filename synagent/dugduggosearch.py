import sseclient
import requests
from urllib.parse import urljoin

MCP_ROOT = "https://agents-mcp-hackathon-mcp-chai1-modal.hf.space"

def stream_mcp():
    sse_url = f"{MCP_ROOT}/gradio_api/mcp/sse"
    print(f"Connecting to MCP via SSE at {sse_url}...")

    session = requests.Session()
    response = session.get(sse_url, stream=True, timeout=60)

    if response.status_code != 200:
        print("❌ Failed to connect to MCP:", response.status_code)
        return

    client = sseclient.SSEClient(response)
    print("Receiving response from MCP...\n")

    for event in client.events():
        data = event.data.strip()
        print("→", data)

        # Redirect to session-specific stream
        if data.startswith("/gradio_api/mcp/messages/"):
            session_url = urljoin(MCP_ROOT, data)
            print(f"🔄 Reconnecting to session SSE stream: {session_url}")

            # Reuse the same session (carries over cookies, headers)
            session_response = session.get(session_url, stream=True, timeout=120)

            if session_response.status_code != 200:
                print("❌ Failed to connect to session stream:", session_response.status_code)
                print("Response content:", session_response.text)
                return

            session_client = sseclient.SSEClient(session_response)
            for session_event in session_client.events():
                if session_event.data.strip() == "[DONE]":
                    print("✅ Done.")
                    return
                print("📨", session_event.data)
            break

if __name__ == "__main__":
    stream_mcp()
