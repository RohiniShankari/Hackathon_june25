import os
import boto3
from botocore.config import Config
from langchain_aws import ChatBedrock

# Setup Bedrock client
config = Config(
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

client = boto3.client(
    'bedrock-runtime',
    region_name="us-east-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    config=config
)

# Create Claude LLM wrapper for LangChain
llm = ChatBedrock(
    model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    model_kwargs={"temperature": 0},
    client=client
)
# llm= llm.with_structured_output