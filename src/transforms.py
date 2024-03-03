import torch
from torch import Tensor

from typing import Any, Tuple, Dict, Optional, Iterable, Callable, List
from array import array
from nearest_key_lookup import NearestKeyLookup
from tokenizers import KeyboardTokenizerv1, CharLevelTokenizerv2


DatasetEl = Tuple[array, array, array, str, Optional[str]]

# class NearestKeyLookupTransform:
#     def __init__(self, grid_name_to_nk_lookup: Dict[str, NearestKeyLookup]) -> None:
#         self.grid_name_to_nk_lookup = grid_name_to_nk_lookup

#     def __call__(self, data: DatasetEl) -> str:
#         X, Y, _, grid_name, _ = data
#         nearest_key_lookup = self.grid_name_to_nk_lookup[grid_name]
#         return nearest_key_lookup.get_nearest_kb_label(X, Y)


# class Compose:
#     def __init__(self, transforms_lst: Iterable[Callable]) -> None:
#         self.transforms_lst = transforms_lst
    
#     def __call__(self, data: Any) -> Any:
#         for transform in self.transforms_lst:
#             data = transform(data)
#         return data
    

# class KeyboardLabelTokenGetter:
#     def __init__(self, tokenizer: KeyboardTokenizerv1) -> None:
#         self.tokenizer = tokenizer

#     def __call__(self, char: str):
#         return self.tokenizer.get_token(char)


# class ComposeTuple:
#     def __init__(self, transforms_lst: Iterable[Callable]) -> None:
#         self.transforms_lst = transforms_lst
    
#     def __call__(self, data: Any) -> Any:
#         result = []
#         for transform in self.transforms_lst:
#             result.append(transform(data))
#         return result


def get_dx_dt(X: Tensor,
              T: Tensor) -> Tensor:
    """
    Calculates dx/dt for a list of x coordinates and a list of t coordinates.

    Arguments:
    ----------
    X : Tensor
        x (position) coordinates.
    T : Tensor
        T[i] = time (ms) from swipe start corresponding to X[i].
    """
    # X, T = (Tensor(arr) for arr in (X, T))
    dx_dt = torch.zeros_like(X)
    # dx_dt[1:-1] = (X[2:] - X[:-2]) / (T[2:] - T[:-2])
    dx_dt[1:len(X)-1] = (X[2:len(X)] - X[:len(X)-2]) / (T[2:len(X)] - T[:len(X)-2])

    # Example:
    # x0 x1 x2 x3
    # t0 t1 t2 t3
    # dx_dt[0] = 0
    # dx_dt[1] = (x2 - x0) / (t2 - t0)
    # dx_dt[2] = (x3 - x1) / (t3 - t1)
    # dx_dt[3] = 0

    # if True in torch.isnan(dx_dt):
    #     print(dx_dt)
    #     raise ValueError("dx_dt contains NaNs")

    return dx_dt


class EncoderFeaturesGetter:
    def __init__(self, 
                 grid_name_to_nk_lookup: Dict[str, NearestKeyLookup],
                 grid_name_to_wh: Dict[str, Tuple[int, int]],
                 kb_tokenizer: KeyboardTokenizerv1,
                 include_time: bool,
                 include_velocities: bool,
                 include_accelerations: bool
                 ) -> None:
        if include_accelerations and not include_velocities:
            raise ValueError("Accelerations are supposed \
                             to be an addition to velocities. Add velocities.")
        
        self.grid_name_to_nk_lookup = grid_name_to_nk_lookup
        self.grid_name_to_wh = grid_name_to_wh
        self.kb_tokenizer = kb_tokenizer
        self.include_time = include_time
        self.include_velocities = include_velocities
        self.include_accelerations = include_accelerations
    
    def _get_kb_tokens(self, X, Y, grid_name) -> Tensor:
        nearest_key_lookup = self.grid_name_to_nk_lookup[grid_name]
        kb_labels = [nearest_key_lookup(x, y) for x, y in zip(X, Y)]
        kb_tokens = [self.kb_tokenizer.get_token(label) for label in kb_labels]
        kb_tokens = Tensor(kb_tokens, dtype=torch.int64)
        return kb_tokens
    
    def _get_traj_feats(self, X: Tensor, Y: Tensor, T: Tensor, 
                        grid_name: str) -> Tensor:
        traj_feats = [X, Y]
        if self.include_time:
            traj_feats.append(T)

        if self.include_velocities:
            dx_dt = get_dx_dt(X, T)
            dy_dt = get_dx_dt(Y, T)
            traj_feats.extend([dx_dt, dy_dt])
        
        if self.include_accelerations:
            d2x_dt2 = get_dx_dt(dx_dt, T)
            d2y_dt2 = get_dx_dt(dy_dt, T)
            traj_feats.extend([d2x_dt2, d2y_dt2])
        
        traj_feats = torch.cat(
            traj_feats,
            axis = 1
        )

        width, height = self.grid_name_to_wh[grid_name]
        traj_feats[:, 0] = traj_feats[:, 0] / width
        traj_feats[:, 1] = traj_feats[:, 1] / height

        return traj_feats


    def __call__(self, X: Iterable, Y: Iterable,
                 T: Iterable, grid_name: str) -> Tuple[Tensor, Tensor]:
        X, Y, T = (Tensor(arr) for arr in (X, Y, T))
        traj_feats = self._get_traj_feats(X, Y, T, grid_name)
        kb_tokens = self._get_kb_tokens(X, Y, grid_name)
        return traj_feats, kb_tokens



class TransformerInputGetter:
    def __init__(self, 
                 grid_name_to_nk_lookup: Dict[str, NearestKeyLookup],
                 grid_name_to_wh: Dict[str, Tuple[int, int]],
                 kb_tokenizer: KeyboardTokenizerv1,
                 word_tokenizer: CharLevelTokenizerv2,
                 include_time: bool,
                 include_velocities: bool,
                 include_accelerations: bool
                 ) -> None:
        self.get_encoder_feats = EncoderFeaturesGetter(
            grid_name_to_nk_lookup, grid_name_to_wh, kb_tokenizer,
            include_time, include_velocities, include_accelerations)
        self.word_tokenizer = word_tokenizer
    
    def get_decoder_in_and_out(self, tgt_word):
        # <sos>, token1, token2, ... token_n, <eos>
        tgt_token_seq: List[int] = self.word_tokenizer.encode(tgt_word)
        tgt_token_seq = torch.tensor(tgt_token_seq, dtype = torch.int64)

        decoder_in = tgt_token_seq[:-1]
        decoder_out = tgt_token_seq[1:]
        return decoder_in, decoder_out


    def __call__(self, data: DatasetEl):
        X, Y, T, grid_name, tgt_word = data
        traj_feats, kb_tokens = self.get_encoder_feats(X, Y, T, grid_name)
        decoder_in, decoder_out = self.get_decoder_in_and_out(tgt_word)
        return (traj_feats, kb_tokens, decoder_in), decoder_out
