from pydantic import BaseModel, Field, field_validator
from rdkit import Chem
from typing import Dict, Optional, List

class CompoundInfo(BaseModel):
    """Pydantic model for holding compound information."""
    name: str = Field(..., description="The name of the compound")
    smiles: str = Field(..., description="The SMILES representation")
    description: str = Field(..., description="Short description of the compound focusing on its chemical properties")

    @field_validator("smiles")
    def validate_smiles(cls, v: str) -> str:
        """Validate the SMILES string."""
        if Chem.MolFromSmiles(v) is None:
            raise ValueError(f"Invalid SMILES: {v}")
        return v

from pydantic import RootModel  # This is new in Pydantic v2

# First, define the inner structure
class EnzymeEntry(BaseModel):
    name: str
    sequence: str
    reaction_smarts: str
# Now use RootModel for the dict
class EnzymeInfo(RootModel[Dict[str, EnzymeEntry]]):
    pass
from langchain.output_parsers import PydanticOutputParser


class ReactionConditionInfo(BaseModel):
    """Model to store recommended reaction conditions."""
    catalyst: str = Field(..., description="Recommended catalyst for the reaction")
    reagent: str = Field(..., description="Recommended reagent or activating agent")
    solvent: str = Field(..., description="Recommended solvent or solvent system")
condition_parser = PydanticOutputParser(pydantic_object=ReactionConditionInfo)
class PatentEntry(BaseModel):
    patent_number: str = Field(..., description="Patent number or short title")
    link: str = Field(..., description="URL of the patent")
    reason: str = Field(..., description="Why this patent might be relevant")

class PatentSearchResults(BaseModel):
    results: List[PatentEntry]

patent_parser = PydanticOutputParser(pydantic_object=PatentSearchResults)