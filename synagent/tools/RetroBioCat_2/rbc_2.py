import inspect,os
path = inspect.getfile(inspect.currentframe())
# print(path)
dir_path = os.path.dirname(os.path.abspath(path))
import sys
sys.path.append(dir_path)
from rbc2 import RetroBioCatExpander
def get_bio_catalyzed_reactions(
    target_mol: str,
    ) -> list:
    """
    Get the reactions for a target molecule using RetroBioCat.
    
    Args:
        target_mol (str): The SMILES representation of the target molecule.
        
    Returns:
        list: A list of reaction objects containing the proposed reactions.
    """
    expander = RetroBioCatExpander()
    reactions = expander.get_reactions(target_mol)

    # print the reaction smiles for the proposed reactions
    rxn_smiles = [rxn.reaction_smiles() for rxn in reactions]

    # Print all the precedents associated with the proposed reactions
    details=[precedent.data for rxn in reactions for precedent in rxn.precedents]
    reactions=dict(zip(rxn_smiles, details))
    
    return reactions
