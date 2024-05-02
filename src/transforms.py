import torch
from torch import Tensor

from typing import Tuple, Dict, Optional, Iterable, List, Callable
from array import array
from nearest_key_lookup import NearestKeyLookup
from distances_lookup import DistancesLookup
from ns_tokenizers import KeyboardTokenizerv1, CharLevelTokenizerv2
from dataset import RawDatasetEl 


GetItemTransformInput = Tuple[array, array, array, str, Optional[str], array]
FullTransformResultType = Tuple[Tuple[Tensor, Tensor, Tensor], Tensor]


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


class TrajFeatsGetter:
    def __init__(self, 
                 grid_name_to_wh: Dict[str, Tuple[int, int]],
                 include_time: bool,
                 include_velocities: bool,
                 include_accelerations: bool
                 ) -> None:
        if include_accelerations and not include_velocities:
            raise ValueError("Accelerations are supposed \
                             to be an addition to velocities. Add velocities.")

        self.grid_name_to_wh = grid_name_to_wh
        self.include_time = include_time
        self.include_velocities = include_velocities
        self.include_accelerations = include_accelerations

    def __call__(self, X: Tensor, Y: Tensor, T: Tensor, grid_name: str) -> Tensor:
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
            [traj_feat.reshape(-1, 1) for traj_feat in traj_feats],
            axis = 1
        )

        width, height = self.grid_name_to_wh[grid_name]
        traj_feats[:, 0] = traj_feats[:, 0] / width
        traj_feats[:, 1] = traj_feats[:, 1] / height

        return traj_feats
    

class KbTokensGetter:
    def __init__(self, 
                 grid_name_to_nk_lookup: Dict[str, NearestKeyLookup],
                 kb_tokenizer: KeyboardTokenizerv1,
                 return_tensor: bool,
                 dtype: torch.dtype = torch.int32,
                 input_to_int: bool = True
                 ) -> None:
        self.dtype = dtype
        self.input_to_int = input_to_int
        self.return_tensor = return_tensor
        self.grid_name_to_nk_lookup = grid_name_to_nk_lookup
        self.kb_tokenizer = kb_tokenizer
    
    def __call__(self, X: Iterable, Y: Iterable, grid_name: str
                 ) -> Tensor:
        nearest_key_lookup = self.grid_name_to_nk_lookup[grid_name]
        caster = int if self.input_to_int else lambda x: x
        kb_labels = [nearest_key_lookup(caster(x), caster(y)) 
                     for x, y in zip(X, Y)]
        kb_tokens = [self.kb_tokenizer.get_token(label) for label in kb_labels]

        if self.return_tensor:
            kb_tokens = torch.tensor(kb_tokens, dtype=self.dtype)
        else:
            kb_tokens = array('B', kb_tokens)

        return kb_tokens
    

class DistancesGetter:
    def __init__(self, 
                 grid_name_to_dists_lookup: Dict[str, DistancesLookup],
                 dtype: torch.dtype = torch.float32
                 ) -> None:
        self.grid_name_to_dists_lookup = grid_name_to_dists_lookup
        self.dtype = dtype

    def __call__(self, X: Iterable, Y: Iterable, grid_name: str
                 ) -> Tensor:
        dl_lookup = self.grid_name_to_dists_lookup[grid_name]
        distances = dl_lookup.get_distances_for_full_swipe_v2(X, Y)
        distances = torch.tensor(distances, dtype=self.dtype)
        return distances


def weights_function_v1(distances: Tensor) -> Tensor:
    """$$f(x) = \frac{1}{1+e^{\frac{1.8x}{key\_radius} - 4}}$$"""

    sigmoid_input = distances * (-1.8) + 4
    return torch.nn.functional.sigmoid(sigmoid_input)
    
    # return 1 / (1 + torch.exp(1.8 * distances - 4))


class KeyWeightsGetter:
    def __init__(self, 
                 grid_name_to_dists_lookup: Dict[str, DistancesLookup],
                 weights_function: Callable,
                 dtype: torch.dtype = torch.float32,
                 ) -> None:
        self.distances_getter = DistancesGetter(grid_name_to_dists_lookup, dtype)
        self.weights_function = weights_function
        self.dtype = dtype

    def __call__(self, X: Iterable, Y: Iterable, grid_name: str
                 ) -> Tensor:
        distances = self.distances_getter(X, Y, grid_name)
        mask = (distances >= 0).to(torch.int32)
        weights = self.weights_function(distances) * mask
        return weights
        


class EncoderFeaturesGetter:
    def __init__(self, 
                 grid_name_to_nk_lookup: Dict[str, NearestKeyLookup],
                 grid_name_to_wh: Dict[str, Tuple[int, int]],
                 kb_tokenizer: KeyboardTokenizerv1,
                 include_time: bool,
                 include_velocities: bool,
                 include_accelerations: bool,
                 kb_tokens_dtype: torch.dtype = torch.int32,
                 ) -> None:
        self._get_traj_feats = TrajFeatsGetter(
            grid_name_to_wh,
            include_time, include_velocities, include_accelerations)
        
        self._get_kb_tokens = KbTokensGetter(
            grid_name_to_nk_lookup, kb_tokenizer, True, dtype=kb_tokens_dtype)

    def __call__(self, X: array, Y: array,
                 T: array, grid_name: str) -> Tuple[Tensor, Tensor]:
        # Conversion to tensor would lead to indexing error since 
        # tensor(`index_val`) is not a proper index 
        # even if `index_val` is an integer
        kb_tokens = self._get_kb_tokens(X, Y, grid_name)

        X, Y, T = (torch.tensor(arr, dtype=torch.float32) for arr in (X, Y, T))
        traj_feats = self._get_traj_feats(X, Y, T, grid_name)
        
        return traj_feats, kb_tokens
    

class EncoderFeaturesGetter_Weighted:
    def __init__(self,
                 grid_name_to_dist_lookup: Dict[str, DistancesLookup],
                 grid_name_to_wh: Dict[str, Tuple[int, int]],
                 include_time: bool,
                 include_velocities: bool,
                 include_accelerations: bool,
                 weights_func: Callable = weights_function_v1
                 ) -> None:    
        self._get_traj_feats = TrajFeatsGetter(
            grid_name_to_wh, 
            include_time, include_velocities, include_accelerations)
        
        self.get_weights = KeyWeightsGetter(
            grid_name_to_dist_lookup, weights_func)

    def __call__(self, X: array, Y: array,
                    T: array, grid_name: str) -> Tuple[Tensor, Tensor]:
            X, Y, T = (torch.tensor(arr, dtype=torch.float32) for arr in (X, Y, T))
            traj_feats = self._get_traj_feats(X, Y, T, grid_name)
            weights = self.get_weights(X, Y, grid_name)
            return traj_feats, weights


class DecoderInputOutputGetter:
    def __init__(self,
                 word_tokenizer: CharLevelTokenizerv2,
                 dtype: torch.dtype = torch.int32
                 ) -> None:
        self.word_tokenizer = word_tokenizer
        self.dtype = dtype
    
    def __call__(self, tgt_word: str, 
                 ) -> Tuple[Tensor, Tensor]:
        # <sos>, token1, token2, ... token_n, <eos>
        tgt_token_seq: List[int] = self.word_tokenizer.encode(tgt_word)
        tgt_token_seq = torch.tensor(tgt_token_seq, dtype=self.dtype)

        decoder_in = tgt_token_seq[:-1]
        decoder_out = tgt_token_seq[1:]
        return decoder_in, decoder_out


class FullTransform:
    def __init__(self, 
                 grid_name_to_nk_lookup: Dict[str, NearestKeyLookup],
                 grid_name_to_wh: Dict[str, Tuple[int, int]],
                 kb_tokenizer: KeyboardTokenizerv1,
                 word_tokenizer: CharLevelTokenizerv2,
                 include_time: bool,
                 include_velocities: bool,
                 include_accelerations: bool,
                 kb_tokens_dtype: torch.dtype = torch.int32,
                 word_tokens_dtype: torch.dtype = torch.int32,
                 ) -> None:
        self.get_encoder_feats = EncoderFeaturesGetter(
            grid_name_to_nk_lookup, grid_name_to_wh, kb_tokenizer,
            include_time, include_velocities, include_accelerations, 
            kb_tokens_dtype)
        self.get_decoder_in_out = DecoderInputOutputGetter(
            word_tokenizer, dtype = word_tokens_dtype)
    
    def __call__(self, data: RawDatasetEl
                 ) -> FullTransformResultType:
        X, Y, T, grid_name, tgt_word = data
        traj_feats, kb_tokens = self.get_encoder_feats(X, Y, T, grid_name)

        decoder_in, decoder_out = None, None
        if tgt_word is not None:
            decoder_in, decoder_out = self.get_decoder_in_out(tgt_word)
        return (traj_feats, kb_tokens, decoder_in), decoder_out



class TrajFeats_KbWeights_FullTransform:
    def __init__(self,
                 grid_name_to_wh: Dict[str, Tuple[int, int]],
                 grid_name_to_dist_lookup: Dict[str, DistancesLookup],
                 word_tokenizer: CharLevelTokenizerv2,
                 include_time: bool,
                 include_velocities: bool,
                 include_accelerations: bool,
                 weights_func: Callable = weights_function_v1,
                 word_tokens_dtype: torch.dtype = torch.int32,
                ) -> None:
        self.get_encoder_feats = EncoderFeaturesGetter_Weighted(
            grid_name_to_dist_lookup, grid_name_to_wh,
            include_time, include_velocities, include_accelerations,
            weights_func)
        
        self.get_decoder_in_out = DecoderInputOutputGetter(
            word_tokenizer, dtype = word_tokens_dtype)

    def __call__(self, data: RawDatasetEl
                    ) -> FullTransformResultType:
            X, Y, T, grid_name, tgt_word = data
            traj_feats, weights = self.get_encoder_feats(X, Y, T, grid_name)
    
            decoder_in, decoder_out = None, None
            if tgt_word is not None:
                decoder_in, decoder_out = self.get_decoder_in_out(tgt_word)
            return (traj_feats, weights, decoder_in), decoder_out



class TokensTypeCastTransform:
    def __call__(self, data: FullTransformResultType) -> FullTransformResultType:
        (traj_feats, kb_tokens, decoder_in), decoder_out = data
        # Embedding layer accepts int32, but not smaller types
        kb_tokens = kb_tokens.to(torch.int32)
        decoder_in = decoder_in.to(torch.int32)
        # CELoss accepts int64 only
        decoder_out = decoder_out.to(torch.int64)
        return (traj_feats, kb_tokens, decoder_in), decoder_out






############################################################  
# Transformes below were used to make training faster while avoiding RAM 
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
        self.get_kb_tokens = KbTokensGetter(
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
        self.get_kb_tokens = KbTokensGetter(
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
