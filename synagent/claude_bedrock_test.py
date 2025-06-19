import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
import boto3

load_dotenv()  

client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",  
    model_kwargs={"temperature": 0},
    client=client
)

response = llm.invoke("Tell me a joke about artificial intelligence.")
print(response)
