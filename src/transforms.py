import torch

from typing import Any, Tuple, Dict, Optional, Iterable, Callable
from array import array
from nearest_key_lookup import NearestKeyLookup
from tokenizers import KeyboardTokenizerv1


DatasetEl = Tuple[array, array, array, str, Optional[str]]

class NearestKeyLookupTransform:
    def __init__(self, grid_name_to_nk_lookup: Dict[str, NearestKeyLookup]) -> None:
        self.grid_name_to_nk_lookup = grid_name_to_nk_lookup

    def __call__(self, data: DatasetEl) -> str:
        X, Y, _, grid_name, _ = data
        nearest_key_lookup = self.grid_name_to_nk_lookup[grid_name]
        return nearest_key_lookup.get_nearest_kb_label(X, Y)


class Compose:
    def __init__(self, transforms_lst: Iterable[Callable]) -> None:
        self.transforms_lst = transforms_lst
    
    def __call__(self, data: Any) -> Any:
        for transform in self.transforms_lst:
            data = transform(data)
        return data
    

class KeyboardLabelTokenGetter:
    def __init__(self, tokenizer: KeyboardTokenizerv1) -> None:
        self.tokenizer = tokenizer

    def __call__(self, char: str):
        return self.tokenizer.get_token(char)


class ComposeTuple:
    def __init__(self, transforms_lst: Iterable[Callable]) -> None:
        self.transforms_lst = transforms_lst
    
    def __call__(self, data: Any) -> Any:
        result = []
        for transform in self.transforms_lst:
            result.append(transform(data))
        return result


def det_dx_dt(x, t):
    pass


class DerivativesAugmentor:
    def __init__(self, 
                 include_accelerations: bool = True, 
                 include_time: bool = False) -> None:
        self.include_accelerations = include_accelerations
        self.include_time = include_time

    def __call__(self, xyt) -> Any:
        X, Y, T = xyt



class FullFeaturesGetter:
    def __init__(self, 
                 grid_name_to_nk_lookup: Dict[str, NearestKeyLookup],
                 kb_tokenizer: KeyboardTokenizerv1
                 ) -> None:
        self.grid_name_to_nk_lookup = grid_name_to_nk_lookup
        self.kb_tokenizer = kb_tokenizer
    
    def _get_kb_tokens(self, X, Y, grid_name):
        nearest_key_lookup = self.grid_name_to_nk_lookup[grid_name]
        kb_labels = [nearest_key_lookup(x, y) for x, y in zip(X, Y)]
        kb_tokens = [self.kb_tokenizer.get_token(label) for label in kb_labels]
        kb_tokens = torch.tensor(kb_tokens, dtype=torch.int64)
        return kb_tokens

    def __call__(self, input_data):
        X, Y, T, grid_name = input_data
        kb_tokens = self._get_kb_tokens(X, Y, T)
        








        

