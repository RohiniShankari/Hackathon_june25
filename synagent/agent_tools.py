from typing import Dict, Any, List
from langchain.output_parsers import PydanticOutputParser
from rxnmapper import RXNMapper
from backup import SynthesisState
from prompts import compound_prompt, compound_parser
from llm_config import llm

# Placeholder imports for actual tool implementations
from tools.RetroBioCat_2.rbc_2 import get_bio_catalyzed_reactions
from tools.CLAIRE.dev.ec_number_predict import get_ec_numbers_from_rxn
from tools.translate_ import inference as r_smiles_inference
from schemas import CompoundInfo, EnzymeInfo,condition_parser,patent_parser
from prompts import partial_ec_prompt,session_summary_prompt,reaction_condition_prompt,query_prompt, reasoning_prompt
# Mock functions for demonstration purposes
# def get_bio_catalyzed_reactions(smiles: str) -> Dict:
#     print(f"Fetching biocatalyzed reactions for {smiles}")
#     return {"reaction1": "C=C.C=C>>C=CC=C"}

# def get_ec_numbers_from_rxn(reactions: List[str]) -> List:
#     print(f"Fetching EC numbers for {reactions}")
#     return [reactions[0], "1.1.1.1"]

# def r_smiles_inference(smiles: str) -> Dict:
#     print(f"Running other retrosynthesis tools for {smiles}")
#     return {'predictions': ['CCO.CC(=O)O'], 'reactants': 'CC(=O)OCC'}


def extract_compound_info(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts compound information from the state using the LLM and updates the state with the compound info."""
    compound_name = state.get("target_mol")
    print('Extracting compound info for:', compound_name)
    formatted = compound_prompt.format_messages(
        compound_name=compound_name,
        format_instructions=compound_parser.get_format_instructions()
    )
    response = llm.invoke(formatted)
    result = compound_parser.parse(response.content if hasattr(response, "content") else response)
    state["compound_info"] = result.model_dump()
    state['target_mol_smiles'] = result.smiles
    print(f"Extracted compound info: {state['compound_info']}")
    return state

def get_bio_synthesis_reactions(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetches biocatalyzed reactions for the target molecule's SMILES from the RBC-2 tool.
    Updates the state with the predicted reactions."""
    print("Getting bio-catalyzed reactions for:", state.get('target_mol_smiles'))
    target_mol_smiles = state.get('target_mol_smiles', '')
    if not target_mol_smiles:
        raise ValueError("Target molecule is required to get reactions.")
    reactions = get_bio_catalyzed_reactions(target_mol_smiles)
    print("Predicted reactions:", reactions)
    state['predicted_reactions'] = list(reactions.keys())
    return state

def retrieve_ec_numbers(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieves EC numbers for the selected reactions from the current synthesis state.
    Updates the state with the predicted EC numbers."""
    reactions_to_process = state.get('selected_reactions') or state.get('predicted_reactions', [])
    if not reactions_to_process:
        raise ValueError("No reactions selected for EC number prediction.")
    
    ec_numbers = get_ec_numbers_from_rxn(reactions_to_process)
    
    ec_number_dict = {}
    for row in ec_numbers:
        row = list(row)  # convert from np.array to list
        reaction = row[0]
        ecs = row[1:]
        ec_number_dict[reaction] = ecs
    
    state['predicted_ec_numbers'] = ec_number_dict
    return state

def run_other_retro_tools(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs other retrosynthesis tools to predict reactions for the target molecule's SMILES.
    Updates the state with the predicted reactions."""
    retro_results = r_smiles_inference(state['target_mol_smiles'])
    rxn_smiles = [pred + '>>' + retro_results['reactants'] for pred in retro_results['predictions']]
    state['predicted_reactions'] = rxn_smiles
    state["messages"].append({"tool": "other_tools", "note": "Ran non-biocatalytic retrosynthesis"})
    return state

def get_atom_mapping(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieves atom mappings for the predicted reactions using RXNMapper.
    Updates the state with the atom mappings."""
    rxn_mapper = RXNMapper()
    results = rxn_mapper.get_attention_guided_atom_maps(state.get('predicted_reactions', []))
    state['atom_mappings'] = results
    return state
def select_reactions_for_enzyme_prediction(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prompts the user to select reactions for enzyme prediction from the predicted reactions.
    Updates the state with the selected reactions."""
    reactions = state.get("predicted_reactions", [])
    if not reactions:
        raise ValueError("No predicted reactions found.")

    print("\nHere are the predicted reactions:\n")
    for idx, rxn in enumerate(reactions):
        print(f"{idx}: {rxn}")
    
    user_input = input("\nEnter the indices of reactions (comma-separated) to predict enzymes for (or type 'all'): ").strip().lower()
    
    if user_input == "all":
        selected_indices = list(range(len(reactions)))
    else:
        try:
            selected_indices = [int(i) for i in user_input.split(",") if i.strip().isdigit()]
        except ValueError:
            raise ValueError("Invalid input. Please enter valid indices separated by commas or type 'all'.")

    selected_reactions = [reactions[i] for i in selected_indices]
    state["selected_reactions"] = selected_reactions
    return state

enzyme_parser = PydanticOutputParser(pydantic_object=EnzymeInfo)

def retrieve_enzymes_from_partial_ec(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieves enzyme information based on the predicted EC numbers from the state.
    Updates the state with the predicted enzymes."""
    ec_dict = state.get("predicted_ec_numbers", {})
    if not ec_dict:
        raise ValueError("No EC numbers found.")

    # Flatten and keep format like EC:3.5.5/0.9989
    ec_candidates = []
    # for parent, children in ec_dict.items():
    #     ec_candidates.append(parent)
    #     ec_candidates.extend(children)

    # unique_ecs = list(set(ec_candidates))
    # print('unique_ecs:', unique_ecs)
    # print(f"Unique EC numbers to process: {unique_ecs}")
    formatted_prompt = partial_ec_prompt.format(ec_number_list=", ".join(ec_dict))

    response = llm.invoke([("human", formatted_prompt)])
    content = response.content.strip()
    print(content)
    try:
        enzyme_info = enzyme_parser.parse(content)
    except Exception as e:
        raise ValueError(f"Failed to parse enzyme info: {e}\nLLM content: {content}")
    
    state["predicted_enzymes"] = enzyme_info
    print("Predicted enzymes:", enzyme_info.model_dump())
    return state
import json

def summarize_and_restart(state: Dict[str, Any]) -> Dict[str, Any]:
    """ Generates a summary of the retrosynthesis session and restarts the state."""

    print("-"*30)
    print("state:", str(state))
    print("-"*30)
    print("\n📋 Generating summary of your retrosynthesis session...\n")

    # Prepare input for LLM
    input_json = json.dumps({
        "compound_info": state.get("compound_info"),
        "target_mol_smiles": state.get("target_mol_smiles"),
        "biocat_preference": state.get("biocat_preference"),
        "predicted_reactions": state.get("predicted_reactions"),
        "predicted_ec_numbers": state.get("predicted_ec_numbers"),
        "enzyme_predictions": state.get("enzyme_predictions", []),
        "atom_mappings": state.get("atom_mappings", []),
        "reaction_conditions": state.get("reaction_conditions", {}),
        "patents": state.get("patents", []),
        "messages": state.get("messages", []),
        "ec_numbers": state.get("ec_numbers", [])
    }, indent=2)

    prompt = session_summary_prompt.format(state_json=input_json)
    response = llm.invoke([("human", prompt)])
    summary_text = response.content.strip()
    state["summary"] = summary_text
    with open("session_summary.txt", "w") as f:
        f.write(summary_text)
    print("🧪 Retrosynthesis Summary:")
    print(summary_text)
    print("\n🔁 Restarting...\n")

    return state
def recommend_reaction_conditions(state: Dict[str, Any]) -> Dict[str, Any]:
    print(state)
    rxn_smiles = state.get("predicted_reactions")  # this should be generated by your retrosynthesis tools
    print("Recommending conditions for:", rxn_smiles)

    formatted = reaction_condition_prompt.format_messages(reaction_smiles=rxn_smiles)
    response = llm.invoke(formatted)

    # Safe fallback to content field if using LangChain's LLM response
    content = response.content if hasattr(response, "content") else response
    print("LLM response for reaction conditions:", content)
    result = condition_parser.parse(content)
    state["reaction_conditions"] = result.model_dump()
    print("Recommended conditions:", state["reaction_conditions"])
    return state


from langchain_core.tools import tool

# @tool
# def reaction_to_fasta(reaction: str, state: SynthesisState) -> str:
#     """
#     Extracts the actual FASTA sequence of a predicted enzyme that catalyzes the given reaction
#     from the current synthesis state.
#     """
#     # Ensure required data exists
#     predicted_reactions = state.get("predicted_reactions")
#     predicted_enzymes = state.get("predicted_enzymes")

#     if not predicted_reactions or not predicted_enzymes:
#         return "No predicted reactions or enzymes available."

#     try:
#         # Find the index of the reaction in the predicted list
#         index = predicted_reactions.index(reaction)
#         enzyme = predicted_enzymes[index]

#         if not enzyme or "sequence" not in enzyme:
#             return f"No valid sequence found for the enzyme catalyzing: {reaction}"

#         return enzyme["sequence"].strip()
    
#     except ValueError:
#         return f"Reaction not found in predicted list: {reaction}"
#     except IndexError:
#         return f"No enzyme found at the matching index for reaction: {reaction}"
from langchain_core.tools import tool
from duckduckgo_search import DDGS

# def search_patents(state: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Searches for patents or research articles related to the predicted reactions and enzymes.
#     Updates the state with the search results.
#     """
#     try:
#         # Step 1: Generate the query using the LLM
#         from json import dumps
#         state_json = dumps({
#             "reactions": state.get("predicted_reactions", []),
#             "enzymes": state.get("predicted_enzymes", []),
#             "conditions": state.get("reaction_conditions", {})
#         }, indent=2)

#         query_response = llm.invoke(query_prompt.format_messages(state_json=state_json))
#         query = query_response.content.strip()
#         print("🔍 LLM-generated search query:", query)

#         # Step 2: Search using DuckDuckGo
#         with DDGS() as ddgs:
#             raw_results = ddgs.text(query, max_results=5)

#         # Format raw results for LLM
#         raw_results_cleaned = "\n".join(
#             f"- Title: {r.get('title')}\n  URL: {r.get('href')}\n  Snippet: {r.get('body')}"
#             for r in raw_results
#         )
#         print(raw_results_cleaned,raw_results)
#         # Step 3: Let LLM extract and reason about results
#         reasoning_response = llm.invoke(reasoning_prompt.format_messages(
#             query=query,
#             raw_results=raw_results_cleaned
#         ))
#         parsed = patent_parser.parse(reasoning_response.content)

#         # Step 4: Store in state
#         state["patent_search_results"] = parsed.model_dump()
#         print("✅ Structured patent search results stored.")
#         return state

#     except Exception as e:
#         state["patent_search_results"] = {"error": str(e)}
#         return state

from typing import Dict, Any
from json import dumps
from langchain_core.tools import tool
from serpapi import GoogleSearch
import os

# SerpAPI patent search tool
@tool
def serpapi_patent_search(query: str) -> list[dict]:
    """
    Uses SerpAPI to search for patent-related information based on the given query.
    """
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("organic_results", [])

# Main function to use inside LangGraph agent
def search_patents(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Searches for patents or research articles related to the predicted reactions and enzymes.
    Updates the state with the search results.
    """
    try:
        # Step 1: Generate the query using the LLM
        state_json = dumps({
            "reactions": state.get("predicted_reactions", []),
            "enzymes": state.get("predicted_enzymes", []),
            "conditions": state.get("reaction_conditions", {})
        }, indent=2)

        query_response = llm.invoke(query_prompt.format_messages(state_json=state_json))
        query = query_response.content.strip()
        print("🔍 LLM-generated search query:", query)

        # Step 2: Search patents using SerpAPI
        results = serpapi_patent_search.invoke({"query": query})
        
        if not results:
            raise Exception("No results returned by SerpAPI.")

        # Format raw results for LLM
        raw_results_cleaned = "\n".join(
            f"- Title: {r.get('title')}\n  URL: {r.get('link')}\n  Snippet: {r.get('snippet')}"
            for r in results
        )

        print("📄 Raw Search Results:\n", raw_results_cleaned)

        # Step 3: Let LLM extract and reason about results
        reasoning_response = llm.invoke(reasoning_prompt.format_messages(
            query=query,
            raw_results=raw_results_cleaned
        ))
        print(reasoning_response.content)
        parsed = patent_parser.parse(reasoning_response.content)
        

        # Step 4: Store in state
        state["patent_search_results"] = parsed.model_dump()
        print("✅ Structured patent search results stored.")
        return state

    except Exception as e:
        state["patent_search_results"] = {"error": str(e)}
        print("❌ Error in search_patents:", e)
        return state

