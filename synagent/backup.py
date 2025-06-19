import os
from typing_extensions import TypedDict, List, Dict, Any, Optional
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from rdkit import Chem
import boto3
from botocore.config import Config
# LangGraph specific components (to integrate the tools into a graph later)
from langgraph.graph import StateGraph, START, END

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

# Tool 1: Extract compound information using Claude + LangChain output parser
class CompoundInfo(BaseModel):
    name: str = Field(..., description="The name of the compound")
    smiles: str = Field(..., description="The SMILES representation")
    description: str = Field(..., description="Short description of the compound")

    @field_validator("smiles")
    def validate_smiles(cls, v):
        if Chem.MolFromSmiles(v) is None:
            raise ValueError(f"Invalid SMILES: {v}")
        return v

compound_parser = PydanticOutputParser(pydantic_object=CompoundInfo)
compound_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a chemistry assistant that returns JSON with name, SMILES, and a short description of a compound."),
    ("human", "Give compound info about {compound_name}.\n{format_instructions}")
])
from langchain_core.prompts import PromptTemplate

biocat_validation_prompt = PromptTemplate.from_template(
    """You are helping decide whether the user wants a biocatalytic synthesis pathway.

User response: "{user_input}"

Answer with only one word: "yes" or "no"."""
)
def ask_biocatalysis_preference(state: Dict[str, Any]) -> Dict[str, Any]:
    """ Ask the user if they want to explore a biocatalytic synthesis pathway."""   
    raw_input_str = input("Do you want to explore a biocatalytic synthesis pathway? (you can type anything): ").strip()
    
    formatted_prompt = biocat_validation_prompt.format(user_input=raw_input_str)
    response = llm.invoke([("human", formatted_prompt)])
    
    normalized = response.content.strip().lower()
    
    if "yes" in normalized:
        state["biocat_preference"] = True
    elif "no" in normalized:
        state["biocat_preference"] = False
    else:
        raise ValueError(f"Ambiguous response: {normalized}")
    
    # Optional: Keep record
    state["messages"].append({
        "type": "user_input",
        "raw_response": raw_input_str,
        "validated_response": normalized
    })
    return state
def route_biocatalysis(state: Dict[str, Any]) -> str:
    print(f"Routing based on biocatalysis preference: {state.get('biocat_preference')}")
    return "bio_reactions" if state.get("biocat_preference") else "other_tools"

def extract_compound_info(state: Dict[str, Any]) -> Dict[str, Any]:
    compound_name = state.get("target_mol")
    formatted = compound_prompt.format_messages(
        compound_name=compound_name,
        format_instructions=compound_parser.get_format_instructions()
    )
    response = llm.invoke(formatted)
    result = compound_parser.parse(response.content if hasattr(response, "content") else response)
    state["compound_info"] = result.model_dump()
    state['target_mol_smiles']=result.smiles
    print(f"Extracted compound info: {state['compound_info']}")
    return state

# Tool 2: Bio-catalyzed reaction prediction
from tools.RetroBioCat_2.rbc_2 import get_bio_catalyzed_reactions

def get_bio_synthesis_reactions(state: Dict[str, Any]) -> Dict[str, Any]:
    target_mol_smiles = state.get('target_mol_smiles', '')
    if not target_mol_smiles:
        raise ValueError("Target molecule is required to get reactions.")
    reactions = get_bio_catalyzed_reactions(target_mol_smiles)
    state['predicted_reactions'] = list(reactions.keys())
    return state

# Tool 3: EC number prediction
from tools.CLAIRE.dev.ec_number_predict import get_ec_numbers_from_rxn

def retrieve_ec_numbers(state: Dict[str, Any]) -> Dict[str, Any]:
    predicted_reactions = state.get('predicted_reactions', [])
    if not predicted_reactions:
        raise ValueError("Predicted reactions are required to retrieve EC numbers.")
    ec_numbers = get_ec_numbers_from_rxn(predicted_reactions)
    # Clean up EC numbers
    ec_number_dict = {}
    for i in range(len(ec_numbers)):
        ec_number_dict[ec_numbers[0]]=ec_numbers[1:]
    
    state['predicted_ec_numbers'] = ec_numbers_dict
    return state
def run_other_retro_tools(state: Dict[str, Any]) -> Dict[str, Any]:
    # Placeholder for integration with other retrosynthetic tools
    state["messages"].append({"tool": "other_tools", "note": "Ran non-biocatalytic retrosynthesis"})
    return state

# Define SynthesisState structure
class SynthesisState(TypedDict):
    target_mol: str
    target_mol_smiles: Optional[str]
    predicted_reactions: Optional[List[str]]
    predicted_ec_numbers: Dict[str, List[str]]
    predicted_enzymes: Optional[List[bool]]
    messages: List[Dict[str, Any]]
    compound_info: Optional[Dict[str, Any]]
    biocat_preference: Optional[bool] 

# Later, these functions can be chained into a LangGraph StateGraph
# # using something like:
# graph = StateGraph(SynthesisState)
# # graph.add_node("compound_info", extract_compound_info)
# # graph.add_node("bio_reactions", get_bio_synthesis_reactions)
# # graph.add_node("ec_number", retrieve_ec_numbers)
# # graph.set_entry_point("compound_info")
# # graph.add_edge("compound_info", "bio_reactions")
# # graph.add_edge("bio_reactions", "ec_number")
# # graph.set_finish_point("ec_number")



# graph.add_node("compound_info", extract_compound_info)
# graph.add_node("ask_user", ask_biocatalysis_preference)
# graph.add_node("bio_reactions", get_bio_synthesis_reactions)
# graph.add_node("ec_number", retrieve_ec_numbers)
# graph.add_node("other_tools", run_other_retro_tools)

# # Start → Compound Info → Ask User
# graph.set_entry_point("compound_info")
# graph.add_edge("compound_info", "ask_user")

# # Branch based on preference
# graph.add_conditional_edges("ask_user", route_biocatalysis, {
#     "bio_reactions": "bio_reactions",
#     "other_tools": "other_tools"
# })

# # Continue the bio path
# graph.add_edge("bio_reactions", "ec_number")
# graph.set_finish_point("ec_number")  # End bio path
# graph.set_finish_point("other_tools")
if __name__ == "__main__":
    graph = StateGraph(SynthesisState)

    graph.add_node("get_compound_info", extract_compound_info)
    graph.add_node("ask_user", ask_biocatalysis_preference)
    graph.add_node("bio_reactions", get_bio_synthesis_reactions)
    graph.add_node("ec_number", retrieve_ec_numbers)
    graph.add_node("other_tools", run_other_retro_tools)

    graph.set_entry_point("get_compound_info")
    graph.add_edge("get_compound_info", "ask_user")

    graph.add_conditional_edges("ask_user", route_biocatalysis, {
        "bio_reactions": "bio_reactions",
        "other_tools": "other_tools"
    })
    graph.add_edge("bio_reactions", "ec_number")
    graph.set_finish_point("ec_number")
    graph.set_finish_point("other_tools")

    # Compile the graph
    app = graph.compile()

    # Run it with a sample input
    initial_state = {
        "target_mol": "aspirin",
        "messages": []
    }

    final_state = app.invoke(initial_state)

    from pprint import pprint
    pprint(final_state)
