import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from typing import Tuple, Dict, Optional, Iterable, List, Callable, Union, Any
from array import array

import torch
from torch import Tensor

from nearest_key_lookup import NearestKeyLookup
from distances_lookup import DistancesLookup
from ns_tokenizers import KeyboardTokenizerv1, CharLevelTokenizerv2
from ns_tokenizers import ALL_CYRILLIC_LETTERS_ALPHABET_ORD
from dataset import RawDatasetEl 
from grid_processing_utils import get_gname_to_wh, get_kb_label

from transforms_v2 import (NearestKbTokensGetter, GetItemTransformInput, 
                           FullTransformResultType, DecoderInputOutputGetter,
                           TrajFeatsGetter)


############################################################  
# Transforms below were used to make training faster while avoiding RAM 
# overflow. Probably won't be needed: multiprocessing in dataloader 
# makes them useless.


class KbTokens_InitTransform:
    """
    Converts (X, Y, T, grid_name, tgt_word) into
    (X, Y, T, grid_name, tgt_word, kb_tokens)
    """
    def __init__(self, 
                 grid_name_to_nk_lookup: Dict[str, NearestKeyLookup],
                 kb_tokenizer: KeyboardTokenizerv1
                 ) -> None:
        self.get_kb_tokens = NearestKbTokensGetter(
            grid_name_to_nk_lookup, kb_tokenizer, False)
        
    def __call__(self, data: RawDatasetEl) -> GetItemTransformInput:
        X, Y, T, grid_name, tgt_word = data
        kb_tokens = self.get_kb_tokens(X, Y, grid_name)
        return (X, Y, T, grid_name, tgt_word, kb_tokens)
    

class KbTokens_GetItemTransform:
    """
    Converts (X, Y, T, grid_name, tgt_word, kb_tokens) into
    (traj_feats, kb_tokens, decoder_in), decoder_out
    """
    def __init__(self, 
                 grid_name_to_wh: Dict[str, Tuple[int, int]],
                 word_tokenizer: CharLevelTokenizerv2,
                 include_time: bool,
                 include_velocities: bool,
                 include_accelerations: bool,
                 kb_tokens_dtype: torch.dtype = torch.int32,
                 tgt_word_dtype: torch.dtype = torch.int32
                 ) -> None:
        self.kb_tokens_dtype = kb_tokens_dtype
        self.get_traj_feats = TrajFeatsGetter(
            grid_name_to_wh, include_time, include_velocities, include_accelerations)
        self.get_decoder_in_out = DecoderInputOutputGetter(
            word_tokenizer, dtype=tgt_word_dtype)

    def __call__(self, data: GetItemTransformInput) -> FullTransformResultType:
        X, Y, T, grid_name, tgt_word, kb_tokens = data
        X, Y, T = (torch.tensor(arr, dtype=torch.float32) for arr in (X, Y, T))
        kb_tokens = torch.tensor(kb_tokens, dtype=self.kb_tokens_dtype)
        traj_feats = self.get_traj_feats(X, Y, T, grid_name)
        decoder_in, decoder_out = None, None
        if tgt_word is not None:
            decoder_in, decoder_out = self.get_decoder_in_out(tgt_word)
        return (traj_feats, kb_tokens, decoder_in), decoder_out

   
class TrajFeatsKbTokensTgtWord_InitTransform:
    #! Mabe add store_gname option   
    def __init__(self, 
                 grid_name_to_nk_lookup: Dict[str, NearestKeyLookup],
                 grid_name_to_wh: Dict[str, Tuple[int, int]],
                 kb_tokenizer: KeyboardTokenizerv1,
                 include_time: bool,
                 include_velocities: bool,
                 include_accelerations: bool,
                 ) -> None:
        self.get_kb_tokens = NearestKbTokensGetter(
            grid_name_to_nk_lookup, kb_tokenizer, False)
        self.get_traj_feats = TrajFeatsGetter(
            grid_name_to_wh, include_time, include_velocities, include_accelerations)
        
    def __call__(self, data: RawDatasetEl) -> Tuple[Tensor, array, str]:
        X, Y, T, grid_name, tgt_word = data
        kb_tokens = self.get_kb_tokens(X, Y, grid_name)
        X, Y, T = (torch.tensor(arr, dtype=torch.float32) for arr in (X, Y, T))
        traj_feats = self.get_traj_feats(X, Y, T, grid_name)
        return (traj_feats, kb_tokens, tgt_word)


class TrajFeatsKbTokensTgtWord_GetItemTransform:
    def __init__(self,
                 word_tokenizer: CharLevelTokenizerv2,
                 kb_tokens_dtype: torch.dtype = torch.int32,
                 tgt_word_dtype: torch.dtype = torch.int32) -> None:
        self.kb_tokens_dtype = kb_tokens_dtype
        self.get_decoder_in_out = DecoderInputOutputGetter(
            word_tokenizer, dtype=tgt_word_dtype)
    
    def __call__(self, data: GetItemTransformInput) -> FullTransformResultType:
        traj_feats, kb_tokens, tgt_word = data
        kb_tokens = torch.tensor(kb_tokens, dtype=self.kb_tokens_dtype)
        decoder_in, decoder_out = None, None
        if tgt_word is not None:
            decoder_in, decoder_out = self.get_decoder_in_out(tgt_word)
        return (traj_feats, kb_tokens, decoder_in), decoder_out
