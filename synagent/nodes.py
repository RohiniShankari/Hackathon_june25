import json
from typing import Dict, Any
from llm_config import llm
from prompts import intent_classification_prompt, biocat_validation_prompt, standard_synthesis_prompt
from agent_tools import (
    extract_compound_info,
    get_bio_synthesis_reactions,
    retrieve_ec_numbers,
    run_other_retro_tools,
    get_atom_mapping,
)

def greet_and_route(state: Dict[str, Any]) -> Dict[str, Any]:
    user_input = input("Welcome! How can I help you today? ").strip()
    state["messages"].append({"type": "user_input", "raw_input": user_input})

    prompt = intent_classification_prompt.format(user_input=user_input)
    response = llm.invoke([("human", prompt)])
    reply = json.loads(response.content.strip().lower())

    if reply.get('retrosynthesis'):
        state["target_mol"] = reply['target_molecule']
        print(f"User wants to synthesize: {state['target_mol']}")
        state['next'] = "get_compound_info"
    else:
        print(f"Assistant: {reply['answer']}")
        state['next'] = "greet_and_route"
    return state

def ask_biocatalysis_preference(state: Dict[str, Any]) -> Dict[str, Any]:
    raw_input_str = input("Do you want to explore a biocatalytic synthesis pathway? (yes/no): ").strip()
    formatted_prompt = biocat_validation_prompt.format(user_input=raw_input_str)
    response = llm.invoke([("human", formatted_prompt)])
    normalized = response.content.strip().lower()

    state["biocat_preference"] = "yes" in normalized
    state["messages"].append({"type": "user_input", "raw_response": raw_input_str, "validated_response": normalized})
    return state

def route_biocatalysis(state: Dict[str, Any]) -> str:
    print(f"Routing based on biocatalysis preference: {state.get('biocat_preference')}")
    return "bio_reactions" if state.get("biocat_preference") else "other_tools"
    
def ask_for_standard_synthesis(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asks the user if they want to try the standard synthesis workflow
    if no biocatalytic reactions were found.
    """
    print("\nNo biocatalytic pathways were found for this molecule.")
    raw_input_str = input("Would you like to try the standard chemical synthesis search instead? (yes/no): ").strip()
    
    # Use the LLM to normalize the user's "yes" or "no" response
    formatted_prompt = standard_synthesis_prompt.format(user_input=raw_input_str)
    response = llm.invoke([("human", formatted_prompt)])
    normalized = response.content.strip().lower()

    # Set the 'next' field in the state to guide the graph
    if "yes" in normalized:
        print("Understood. Rerouting to standard synthesis tools...")
        state["next"] = "other_tools"
    else:
        print("Okay, returning to the start.")
        state["next"] = "greet_and_route"
        
    # Append the interaction to messages for logging
    state["messages"].append({
        "type": "user_input", 
        "raw_response": raw_input_str, 
        "validated_response": normalized
    })
    
    return state