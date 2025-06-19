import os
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from rdkit import Chem
import boto3,json
from botocore.config import Config
# LangGraph specific components (to integrate the tools into a graph later)
from agent_tools import reaction_to_fasta
from chai1_tools2 import create_fasta_file,create_json_config,compute_chai1,plot_protein,show_cif_file
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
from langchain_core.prompts import PromptTemplate


intent_classification_prompt = PromptTemplate.from_template(
    """You are an assistant for retrosynthesis and general chemistry questions.

Your task is to analyze the user's input and return a JSON object with two fields:
- "retrosynthesis": true if the user is asking how to make, synthesize, or retrosynthesize a compound; otherwise false.
- "target_molecule": the name of the compound the user wants to make or know about, if clearly mentioned; otherwise null.

Only return a valid JSON object. Do not include any extra commentary.

User input: "{user_input}"

Respond with a JSON object like:
{{"retrosynthesis": true, "target_molecule": "aspirin"}}
or
{{"retrosynthesis": false, "target_molecule": null}}"""
)

def greet_and_route(state: Dict[str, Any]) -> Dict[str, Any]:
    user_input = input("Welcome! How can I help you today? ").strip()
    state["messages"].append({"type": "user_input", "raw_input": user_input})

    prompt = intent_classification_prompt.format(user_input=user_input)
    response = llm.invoke([("human", prompt)])
    reply = response.content.strip().lower()
    reply= json.loads(reply)
    # LLM decision based
    print(reply)
    if reply['retrosynthesis'] == "true" or reply['retrosynthesis'] == True:
        state["target_mol"] = reply['target_molecule']
        print(f"User wants to synthesize: {state['target_mol']}")
        state['next'] = "get_compound_info"
        # return {"next": "get_compound_info", "state": state}
    else:
        print(f"Assistant: {response.content.strip()}")
        state['next'] = "greet_and_route"
        # return {"next": "greet_and_route", "state": state}
    return state
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
    print('Extracting compound info for:', compound_name)
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
    print("Getting bio-catalyzed reactions for:", state.get('target_mol_smiles'))
    target_mol_smiles = state.get('target_mol_smiles', '')
    if not target_mol_smiles:
        raise ValueError("Target molecule is required to get reactions.")
    reactions = get_bio_catalyzed_reactions(target_mol_smiles)
    print("Predicted reactions:", reactions)
    state['predicted_reactions'] = list(reactions.keys())
    return state

# Tool 3: EC number prediction
from tools.CLAIRE.dev.ec_number_predict import get_ec_numbers_from_rxn

def retrieve_ec_numbers(state: Dict[str, Any]) -> Dict[str, Any]:
    print(state)
    predicted_reactions = state.get('predicted_reactions', [])
    if not predicted_reactions:
        raise ValueError("Predicted reactions are required to retrieve EC numbers.")
    ec_numbers = get_ec_numbers_from_rxn(predicted_reactions)
    # Clean up EC numbers
    ec_number_dict = {}
    for i in range(len(ec_numbers)):
        ec_number_dict[ec_numbers[0]]=ec_numbers[1:]
    
    state['predicted_ec_numbers'] = ec_number_dict
    return state
from tools.translate_ import inference as r_smiles_inference
# Tool 4: Other retrosynthetic tools (placeholder)
def run_other_retro_tools(state: Dict[str, Any]) -> Dict[str, Any]:
    # Placeholder for integration with other retrosynthetic tools
    retro_results = r_smiles_inference(state['target_mol_smiles'])
    rxn_smiles = [retro_results['predictions'][i]+'>>'+retro_results['reactants'] for i in range(len(retro_results['predictions']))]
    state['predicted_reactions'] = rxn_smiles
    state["messages"].append({"tool": "other_tools", "note": "Ran non-biocatalytic retrosynthesis"})
    return state
from rxnmapper import RXNMapper
def get_atom_mapping(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get atom mapping for the predicted reactions.
    
    Args:
        state (Dict[str, Any]): The current state containing predicted reactions.
        
    Returns:
        Dict[str, Any]: Updated state with atom mappings.
    """
    rxn_mapper = RXNMapper()
    results = rxn_mapper.get_attention_guided_atom_maps(state.get('predicted_reactions', []))    
    state['atom_mappings'] = results
    return state

def ask_protein_visualization(state: Dict[str, Any]) -> Dict[str, Any]:
    user_input = input("Do you want to visualize the protein structure of any predicted reaction? (yes/no): ").strip().lower()
    state["protein_visualization"] = user_input in ["yes", "y"]
    return state

def choose_predicted_reaction(state: Dict[str, Any]) -> Dict[str, Any]:
    reactions = state.get("predicted_reactions", [])
    print("\nPredicted reactions:")
    for idx, rxn in enumerate(reactions):
        print(f"{idx + 1}. {rxn}")

    choice = int(input("Enter the number of the reaction to visualize (1-based index): ").strip()) - 1
    chosen_rxn = reactions[choice]
    state["selected_reaction"] = chosen_rxn
    return state

from langchain_core.tools import tool
from langchain_core.language_models import ChatModel

# You can use Claude, GPT-4, or any model available





def visualize_protein(state: Dict[str, Any]) -> Dict[str, Any]:
    selected_reaction = state["selected_reaction"]
    print(f"🔬 Visualizing protein for: {selected_reaction}")

    # 🔁 Generate hypothetical FASTA using LLM
    # fasta_sequence = reaction_to_fasta.invoke({"reaction": selected_reaction})
    fasta_sequence = reaction_to_fasta.invoke({"reaction": rxn_str, "state": SynthesisState}) # type: ignore
    print(f"📄 FASTA Sequence generated:\n{fasta_sequence[:60]}...")

    # ✅ Create FASTA file using generated sequence
    fasta_result = create_fasta_file.invoke({
        "file_content": fasta_sequence,
        "filename": "reaction_protein.fasta"
    })

    # ✅ Create JSON config
    config_result = create_json_config.invoke({
        "num_diffn_timesteps": 300,
        "num_trunk_recycles": 3,
        "seed": 42,
        "options": ["ESM_embeddings"],
        "filename": "config_reaction.json"
    })

    # ✅ Run simulation
    result = compute_chai1.invoke({
        "fasta_file_name": fasta_result,
        "config_file_name": config_result
    })

    # ✅ Plot protein
    plot_path = plot_protein.invoke({"result_df": result})

    # ✅ Download CIF file
    top_model_cif = result["data"][0][-1]
    cif_url = f"https://agents-mcp-hackathon-mcp-chai1-modal.hf.space/file={top_model_cif}"
    cif_path = show_cif_file.invoke({"cif_file_url": cif_url})

    # ✅ Update state
    state["protein_plot_path"] = plot_path
    state["cif_path"] = cif_path
    print(f"✅ Protein structure saved to:\n- PDB: {plot_path}\n- CIF: {cif_path}")
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

graph = StateGraph(SynthesisState)

graph.add_node("greet_and_route", greet_and_route)
graph.add_node("get_compound_info", extract_compound_info)
graph.add_node("ask_user", ask_biocatalysis_preference)
graph.add_node("bio_reactions", get_bio_synthesis_reactions)
graph.add_node("ec_number", retrieve_ec_numbers)
graph.add_node("other_tools", run_other_retro_tools)
graph.add_node("get_atom_mapping", get_atom_mapping)
# Routing logic
graph.set_entry_point("greet_and_route")
graph.add_conditional_edges("greet_and_route", lambda x: x["next"], {
    "get_compound_info": "get_compound_info",
    "greet_and_route": "greet_and_route"
})

graph.add_edge("get_compound_info", "ask_user")
graph.add_conditional_edges("ask_user", route_biocatalysis, {
    "bio_reactions": "bio_reactions",
    "other_tools": "other_tools"
})
graph.add_edge("bio_reactions", "ec_number")

# Loop back after finishing
# graph.add_edge("ec_number", "greet_and_route")
graph.add_node("ask_protein", ask_protein_visualization)
graph.add_node("choose_reaction", choose_predicted_reaction)
graph.add_node("visualize_protein", visualize_protein)

# Route depending on user preference
graph.add_edge("ec_number", "ask_protein")  # or "other_tools" → "ask_protein"
graph.add_conditional_edges("ask_protein", lambda s: "choose_reaction" if s.get("protein_visualization") else "greet_and_route", {
    "choose_reaction": "choose_reaction",
    "greet_and_route": "greet_and_route"
})

graph.add_edge("choose_reaction", "visualize_protein")
graph.add_edge("visualize_protein", "greet_and_route")

graph.add_edge("other_tools", "greet_and_route")
if __name__ == "__main__":
    app = graph.compile()
    initial_state = {"messages": []}

    try:
        while True:
            final_state = app.invoke(initial_state)
            initial_state = {"messages": []}  # reset
    except KeyboardInterrupt:
        print("\nSession ended.")