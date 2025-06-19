from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from schemas import CompoundInfo

# Parser for compound information
compound_parser = PydanticOutputParser(pydantic_object=CompoundInfo)

# Prompt for intent classification
intent_classification_prompt = PromptTemplate.from_template(
    """You are an assistant for retrosynthesis and general chemistry questions.

Your task is to analyze the user's input and return a JSON object with the following fields:
- "retrosynthesis": true if the user is asking how to make, synthesize, or retrosynthesize a compound; otherwise false.
- "target_molecule": the name of the compound the user wants to make or know about, if clearly mentioned; otherwise null.
- "answer": if the question is *not* about retrosynthesis, return a short informative response to the question; otherwise null.

Only return a valid JSON object. Do not include any extra commentary.

User input: "{user_input}"

Respond with a JSON object like:
{{"retrosynthesis": true, "target_molecule": "aspirin", "answer": null}}
or
{{"retrosynthesis": false, "target_molecule": null, "answer": "Aspirin is a common pain reliever that inhibits COX enzymes."}}
"""
)


# Prompt for extracting compound info
compound_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a chemistry assistant that returns JSON with name, SMILES, and a short description of a compound."),
    ("human", "Give compound info about {compound_name}.\n{format_instructions}")
])

# Prompt for biocatalysis preference
biocat_validation_prompt = PromptTemplate.from_template(
    """You are helping decide whether the user wants a biocatalytic synthesis pathway.

User response: "{user_input}"

Answer with only one word: "yes" or "no"."""
)
standard_synthesis_prompt = PromptTemplate.from_template(
    """The user was asked if they want to try a standard chemical synthesis route after a biocatalytic one failed.
    
Based on their response, answer with only one word: "yes" or "no".

User response: "{user_input}"
"""
)

partial_ec_prompt = PromptTemplate.from_template(
    """You are a biochemistry assistant. The user provides a list of partial EC numbers in the format "EC:x.y.z/confidence".

Your task is to return a **valid JSON object** where each key is the full EC number with confidence (e.g. "EC:1.1.1/0.987"), and each value is a dictionary containing:
- `"name"`: The most likely enzyme name based on the first three EC digits.
- `"sequence"`: A plausible amino acid sequence in single-letter code (between 20 and 60 residues).
- `"reaction_smarts"`: A representative SMARTS or pseudo-SMARTS string describing the type of reaction this enzyme catalyzes.

If the exact enzyme is unclear, make a biologically plausible guess based on the enzyme class.

Respond ONLY with valid JSON in this format don't generate any other text:

{{
  "EC:1.1.1/0.987": {{
    "name": "Alcohol dehydrogenase",
    "sequence": "MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGQKL",
    "reaction_smarts": "[C:1][O:2][H]>>[C:1]=O.[H][O:2]H"
  }},
  "EC:2.7.1/0.899": {{
    "name": "Hexokinase",
    "sequence": "MKTAYIAKQRQISFVKSHFSRQDILDLWQTVVAIYKEAKK",
    "reaction_smarts": "[C:1][OH]>>[C:1][OPO3H2]"
  }}
}}

Generate results for up to 5 EC numbers.

EC numbers:
{ec_number_list}
"""
)

session_summary_prompt = PromptTemplate.from_template(
    """You are a chemistry assistant summarizing the retrosynthesis session.

Here is the state dictionary of the session:
{state_json}

Write a structured and helpful summary for the user, including:
- The compound name and SMILES
- Whether biocatalysis was used
- Number and type of reactions predicted and their reaction smarts, their properties, feasibility and their reaction smarts and Enzyme sequences and Atom mappings if available
- EC numbers and enzyme sequences and their corresponding reaction smarts if available
- Any tool-based steps used

Respond in clear bullet points or formatted markdown text.
"""
)
reaction_condition_prompt = ChatPromptTemplate.from_template(
    """You are a reaction condition recommendation assistant. who first analyzes the reaction and it's mechanism and then suggests optimal conditions.

Given a chemical reaction described by its reactants and products (in SMILES format), suggest the most suitable:
- catalyst (if applicable),
- reagent or activating agent, and
- solvent.

Respond ONLY with a JSON object using this format do not generate any other text:
{{
  "catalyst": "Pd/C",
  "reagent": "H2",
  "solvent": "ethanol"
}}

Reaction SMILES: {reaction_smiles}
"""
)
query_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a chemoinformatics assistant that finds relevant patent search queries."),
    ("human", """Given the retrosynthesis state below, generate a focused query string that searches for similar patents on Google Patents.

State:
{state_json}

Only return the query string. Include important parts such as reaction SMILES, enzyme names, EC numbers, and conditions if available.""")
])

# ---------------- Prompt for analyzing search results ----------------
reasoning_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a chemoinformatics assistant helping to find similar patents."),
    ("human", """Given the search results and the original query, extract the most relevant information and explain why each link is a good match. Return the results in JSON using this structure:

[
  {{
    "patent_number": "<extracted ID or short title>",
    "link": "<patent URL>",
    "reason": "<why it's relevant>"
  }},
  ...
]

Query: {query}
Results:
{raw_results}
""")
])
