import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator, ValidationError
from rdkit import Chem

# Load env
load_dotenv()

# Create Claude LLM client (NO aws_access_key/secret_key here!)
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"temperature": 0.0},
    region_name=os.getenv("AWS_REGION")
)

# Schema with RDKit validation
class CompoundInfo(BaseModel):
    name: str = Field(..., description="The name of the compound")
    smiles: str = Field(..., description="The SMILES representation")
    description: str = Field(..., description="Short description of the compound")

    @field_validator("smiles")
    def validate_smiles(cls, v):
        if Chem.MolFromSmiles(v) is None:
            raise ValueError(f"Invalid SMILES: {v}")
        return v

# Output parser
parser = PydanticOutputParser(pydantic_object=CompoundInfo)

# Prompt template (Anthropic-compatible)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that returns JSON with compound details."),
    ("human", "Give compound info about {compound_name}.\n{format_instructions}")
])

# List of compounds
compound_names = ["caffeine", "aspirin", "acetaminophen", "sodium chloride"]

# Loop through each
for name in compound_names:
    try:
        formatted_prompt = prompt.format_messages(
            compound_name=name,
            format_instructions=parser.get_format_instructions()
        )
        response = llm.invoke(formatted_prompt)
        parsed = parser.parse(response.content if hasattr(response, "content") else response)
        print("✅", parsed.model_dump())
    except Exception as e:
        print(f"❌ Error for {name}: {e}")
