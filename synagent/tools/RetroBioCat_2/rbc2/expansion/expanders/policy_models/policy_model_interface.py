from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

smarts_dict = Dict[str, List[str]]
metadata_dict = Dict[str, dict]


class PolicyModel(ABC):

    @abstractmethod
    def get_rxns(self, smi: str) -> Tuple[smarts_dict, metadata_dict]:
        pass
