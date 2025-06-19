from typing import  List, Dict, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from nodes import (
    greet_and_route,
    ask_biocatalysis_preference,
    route_biocatalysis,
    ask_for_standard_synthesis
)
from agent_tools import (
    extract_compound_info,
    get_bio_synthesis_reactions,
    retrieve_ec_numbers,
    run_other_retro_tools,
    get_atom_mapping,
    retrieve_enzymes_from_partial_ec,
    summarize_and_restart,
    select_reactions_for_enzyme_prediction,
    recommend_reaction_conditions,
    search_patents,
)

class SynthesisState(TypedDict):
    target_mol: str
    target_mol_smiles: Optional[str]
    predicted_reactions: Optional[List[str]]
    predicted_ec_numbers: Dict[str, List[str]]
    predicted_enzymes: Optional[List[bool]]
    messages: List[Dict[str, Any]]
    compound_info: Optional[Dict[str, Any]]
    biocat_preference: Optional[bool]
    next: Optional[str]
    patents: Optional[List[Dict[str, Any]]]
    atom_mapping: Optional[Dict[str, Any]]
    reaction_conditions: Optional[Dict[str, Any]]
    ec_numbers: Optional[List[str]]
def decide_after_bio_reactions(state: SynthesisState) -> str:
    """
    Checks if biocatalytic reactions were found.
    
    If yes, proceeds to get EC numbers.
    If no, asks the user if they want to try another method.
    """
    if len(state.get("predicted_reactions"))>0:
        print("Biocatalytic reactions found. Proceeding to get EC numbers.")
        return "select_reactions"
    else:
        print("No biocatalytic reactions found.")
        return "ask_for_standard_synthesis"
def create_graph() -> StateGraph:
    graph = StateGraph(SynthesisState)

    # Nodes
    graph.add_node("greet_and_route", greet_and_route)
    graph.add_node("get_compound_info", extract_compound_info)
    graph.add_node("ask_user", ask_biocatalysis_preference)
    graph.add_node("bio_reactions", get_bio_synthesis_reactions)
    graph.add_node("select_reactions", select_reactions_for_enzyme_prediction)  # New node
    graph.add_node("ec_number", retrieve_ec_numbers)
    graph.add_node("enzyme_lookup", retrieve_enzymes_from_partial_ec)
    graph.add_node("other_tools", run_other_retro_tools)
    graph.add_node("get_atom_mapping", get_atom_mapping)
    graph.add_node("ask_for_standard_synthesis", ask_for_standard_synthesis)
    graph.add_node("summarize_and_restart", summarize_and_restart)
    graph.add_node("recommend_conditions", recommend_reaction_conditions)
    graph.add_node("search_patents", search_patents)
    graph.set_entry_point("greet_and_route")

    # Routing logic
    graph.add_conditional_edges("greet_and_route", lambda x: x["next"], {
        "get_compound_info": "get_compound_info",
        "greet_and_route": "greet_and_route"
    })

    graph.add_edge("get_compound_info", "ask_user")

    graph.add_conditional_edges("ask_user", route_biocatalysis, {
        "bio_reactions": "bio_reactions",
        "other_tools": "other_tools"
    })

    graph.add_conditional_edges(
        "bio_reactions",
        decide_after_bio_reactions,
        {
            "select_reactions": "select_reactions",
            "ask_for_standard_synthesis": "ask_for_standard_synthesis"
        }
    )

    # Newly added edge for reaction selection → EC prediction
    graph.add_edge("select_reactions", "ec_number")

    graph.add_conditional_edges(
        "ask_for_standard_synthesis",
        lambda x: x["next"],
        {
            "other_tools": "other_tools",
            "greet_and_route": "greet_and_route"
        }
    )
    graph.add_edge("ec_number", "enzyme_lookup")
    graph.add_edge("enzyme_lookup", "search_patents")
    graph.add_edge("other_tools", "get_atom_mapping")
    graph.add_edge("get_atom_mapping", "recommend_conditions")
    graph.add_edge("recommend_conditions", "search_patents")
    # graph.add_edge("enzyme_lookup","summarize_and_restart")
    graph.add_edge("search_patents", "summarize_and_restart")
    graph.add_edge("summarize_and_restart", "greet_and_route")

    return graph.compile()
