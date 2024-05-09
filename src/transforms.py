"""
The dataset by default returns a tuple of 5 elements:
(X, Y, T, grid_name, tgt_word)

Transforms defined in this module are used 
to convert this tuple a tuple (model_in, model_out)
"""

from typing import Tuple, Dict, Optional, Iterable, List, Callable
from array import array

import torch
from torch import Tensor

from nearest_key_lookup import NearestKeyLookup
from distances_lookup import DistancesLookup
from ns_tokenizers import KeyboardTokenizerv1, CharLevelTokenizerv2
from ns_tokenizers import ALL_CYRILLIC_LETTERS_ALPHABET_ORD
from dataset import RawDatasetEl 
from grid_processing_utils import get_gname_to_wh, get_kb_label


DEFAULT_ALLOWED_KEYS = ALL_CYRILLIC_LETTERS_ALPHABET_ORD
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
        # distances = dl_lookup.get_distances_for_full_swipe_without_map(X, Y)
        distances = dl_lookup.get_distances_for_full_swipe_using_map(X, Y)
        distances = torch.tensor(distances, dtype=self.dtype)
        return distances


def weights_function_v1(distances: Tensor, half_key_diag, bias = 4, scale = 1.8) -> Tensor:
    """
    $$f(x) = \frac{1}{1+e^{\frac{s \cdot x}{key\_radius} - b}}$$
    b = bias = 4
    s = scale = 1.8
    """
    
    # return 1 / (1 + torch.exp(1.8 * distances - 4))

    # Sqrt beacuse currently distances is squared euclidian distance

    #! It may be a good idea to move division by half_key_diag outside
    # this function.  The division is just a scaling of distances
    # so that they are not in pixels but use half_key_diag as a unit. 
    sigmoid_input = distances.sqrt() / half_key_diag * (-scale) + bias
    return torch.nn.functional.sigmoid(sigmoid_input)


def weights_function_v1_softmax(distances: Tensor, half_key_diag, bias = 4, scale = 1.8) -> Tensor:
    mask = torch.isinf(distances)
    sigmoid_input = distances.sqrt() / half_key_diag * (-scale) + bias
    weights = torch.nn.functional.sigmoid(sigmoid_input)
    # -inf to zero out unpresent values and have a sum of one 
    weights.masked_fill_(mask, float('-inf'))
    return torch.nn.functional.softmax(weights, dim=1)


class KeyWeightsGetter:
    def __init__(self, 
                 grid_name_to_dists_lookup: Dict[str, DistancesLookup],
                 grid_name_to_half_key_diag: Dict[str, float],
                 weights_function: Callable,
                 dtype: torch.dtype = torch.float32,
                 ) -> None:
        self.distances_getter = DistancesGetter(grid_name_to_dists_lookup, dtype)
        self.weights_function = weights_function
        self.grid_name_to_half_key_diag = grid_name_to_half_key_diag
        self.dtype = dtype

    def __call__(self, X: Iterable, Y: Iterable, grid_name: str
                 ) -> Tensor:
        distances = self.distances_getter(X, Y, grid_name)
        mask = (distances < 0)
        distances.masked_fill_(mask=mask, value = float('inf'))
        half_key_diag = self.grid_name_to_half_key_diag[grid_name]
        weights = self.weights_function(distances, half_key_diag)
        weights.masked_fill_(mask=mask, value=0)
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




def get_avg_half_key_diag(grid: dict, 
                          allowed_keys: List[str] = tuple(DEFAULT_ALLOWED_KEYS)) -> float:
    hkd_list = []
    for key in grid['keys']:
        label = get_kb_label(key)
        if label not in allowed_keys:
            continue
        hitbox = key['hitbox']
        kw, kh = hitbox['w'], hitbox['h']
        half_key_diag = (kw**2 + kh**2)**0.5 / 2
        hkd_list.append(half_key_diag)
    return sum(hkd_list) / len(hkd_list)

    
def get_gname_to_half_key_diag(gname_to_grid: Dict[str, dict], 
                               allowed_keys: List[str] = tuple(DEFAULT_ALLOWED_KEYS)
                               ) -> Dict[str, float]:
    result = {gname: None for gname in gname_to_grid}
    for gname, grid in gname_to_grid.items():
        result[gname] = get_avg_half_key_diag(grid, allowed_keys)
    return result




class EncoderFeaturesGetter_Weighted:
    def __init__(self,
                 grid_name_to_dist_lookup: Dict[str, DistancesLookup],
                 grid_name_to_grid: Dict[str, dict],
                 include_time: bool,
                 include_velocities: bool,
                 include_accelerations: bool,
                 weights_func: Callable = weights_function_v1,
                 allowed_keys = DEFAULT_ALLOWED_KEYS
                 ) -> None:    
        gname_to_wh = get_gname_to_wh(grid_name_to_grid)
        self._get_traj_feats = TrajFeatsGetter(
            gname_to_wh, 
            include_time, include_velocities, include_accelerations)
        
        gname_to_hkd = get_gname_to_half_key_diag(grid_name_to_grid, 
                                                  allowed_keys)
        self.get_weights = KeyWeightsGetter(
            grid_name_to_dist_lookup, gname_to_hkd, weights_func)

    def __call__(self, X: array, Y: array,
                    T: array, grid_name: str) -> Tuple[Tensor, Tensor]:
            # Conversion to tensor would lead to indexing error since 
            # tensor(`index_val`) is not a proper index 
            # even if `index_val` is an integer
            weights = self.get_weights(X, Y, grid_name)

            X, Y, T = (torch.tensor(arr, dtype=torch.float32) for arr in (X, Y, T))
            traj_feats = self._get_traj_feats(X, Y, T, grid_name)
            
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
                 grid_name_to_grid: Dict[str, dict],
                 grid_name_to_dist_lookup: Dict[str, DistancesLookup],
                 word_tokenizer: CharLevelTokenizerv2,
                 include_time: bool,
                 include_velocities: bool,
                 include_accelerations: bool,
                 weights_func: Callable = weights_function_v1,
                 word_tokens_dtype: torch.dtype = torch.int32,
                ) -> None:
        self.get_encoder_feats = EncoderFeaturesGetter_Weighted(
            grid_name_to_dist_lookup, grid_name_to_grid,
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



###################    util transforms    ###################

class SequentialTransform:
    def __init__(self, transforms) -> None:
        self.transforms = transforms
    
    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data










###################    Full transforms aquiring    ###################


from typing import Callable, Tuple, Optional, List, Dict, Set, Iterable
import json

import numpy as np
from tqdm.auto import tqdm

from grid_processing_utils import get_grid
from nearest_key_lookup import ExtendedNearestKeyLookup



class RandIntToTrajTransform:
    def __init__(self, min_ = -3, max_ = 3) -> None:
        self.min = min_
        self.max = max_
        
    def __call__(self, data):
        X, Y, T, grid_name, tgt_word = data
        X = np.array(X, dtype = int) + np.random.randint(self.min, self.max, (len(X),))
        Y = np.array(Y, dtype = int) + np.random.randint(self.min, self.max, (len(Y),))
        return X, Y, T, grid_name, tgt_word
    



def get_grid(grid_name: str, grids_path: str) -> dict:
    with open(grids_path, "r", encoding="utf-8") as f:
        return json.load(f)[grid_name]



def get_gridname_to_out_of_bounds_coords_dict(
        data_paths: List[str], gridname_to_wh: dict,
        totals: Iterable[Optional[int]] = None
        ) -> Dict[str, Set[Tuple[int, int]]]:
    """
    Returns a dictionary with grid names as keys and lists of out of bounds coordinates as values.
    """
    totals = totals or [None] * len(data_paths)
    
    gname_to_out_of_bounds = {gname: set() for gname in gridname_to_wh.keys()}

    for data_path, total in zip(data_paths, totals):
        with open(data_path, "r", encoding="utf-8") as json_file:
            for line in tqdm(json_file, total=total):
                json_data = json.loads(line)
                curve = json_data['curve']
                grid_name = curve['grid_name']
                w, h = gridname_to_wh[grid_name]
                X, Y = curve['x'], curve['y']
                out_of_bounds = set((x, y) for x, y in zip(X, Y) 
                                    if x < 0 or x >= w or y < 0 or y >= h)
                gname_to_out_of_bounds[grid_name].update(out_of_bounds)
    return gname_to_out_of_bounds



def update_out_of_bounds_with_noise(
    noise_min, noise_max,
    gname_to_out_of_bounds, gridname_to_wh: dict,
    )-> Dict[str, Set[Tuple[int, int]]]:
    
    assert noise_min <= 0
    assert noise_max >= 0
    
    additional_out_of_bounds = {gname: set() for gname in gridname_to_wh.keys()}
    
    for gname in gname_to_out_of_bounds.keys():
        w, h = gridname_to_wh[gname]
        
        for x, y in gname_to_out_of_bounds[gname]:
            for i in range(noise_min, noise_max+1):
                for j in range(noise_min, noise_max+1):
                    if x+i < 0 or x+i >= w or y+j < 0 or y+j >=h: 
                        additional_out_of_bounds[gname].add((x+i, y+j))
        
        for x in range(noise_min, w+noise_max+1):
            for y in range(noise_min, 0):
                additional_out_of_bounds[gname].add((x, y))
        
        for x in range(noise_min, w+noise_max+1):
            for y in range(h+1, h+noise_max+1):
                additional_out_of_bounds[gname].add((x, y))
        
        for x in range(w, w+noise_max+1):
            for y in range(0, h+1):
                additional_out_of_bounds[gname].add((x, y))
        
        for x in range(noise_min, 0):
            for y in range(0, h+1):
                additional_out_of_bounds[gname].add((x, y))
                
        gname_to_out_of_bounds[gname].update(additional_out_of_bounds[gname])
        
    return gname_to_out_of_bounds
        


def get_traj_and_nearest_key_transform(grid_name: str,
                                       grid: dict,
                                       char_tokenizer: CharLevelTokenizerv2,
                                       kb_tokenizer: KeyboardTokenizerv1,
                                       gname_to_wh: Dict[str, Tuple[int, int]],
                                       uniform_noise_range: int = 0,
                                       ds_paths_list: Optional[List[str]] = None,
                                       totals: Tuple[Optional[int], Optional[int]] = (None, None),
                                       ) -> Callable:
    if ds_paths_list is not None:
        print("Accumulating out-of-bounds coordinates...")
        gname_to_out_of_bounds = get_gridname_to_out_of_bounds_coords_dict(
            ds_paths_list, 
            gridname_to_wh = gname_to_wh,
            totals=totals
        )

        print("augmenting gname_to_out_of_bounds")
        gname_to_out_of_bounds = update_out_of_bounds_with_noise(
            noise_min = -uniform_noise_range, noise_max=uniform_noise_range+1,
            gname_to_out_of_bounds = gname_to_out_of_bounds, gridname_to_wh = gname_to_wh,
        )
    else:
        print("Warning: no ds_paths_list provided. gname_to_out_of_bounds will be empty.")
        gname_to_out_of_bounds = {gname: set() for gname in gname_to_wh.keys()}


    print("Creating ExtendedNearestKeyLookups...")
    gridname_to_nkl = {
        grid_name: ExtendedNearestKeyLookup(
            grid, ALL_CYRILLIC_LETTERS_ALPHABET_ORD,
            gname_to_out_of_bounds[grid_name]
        )
    }

    full_transform = FullTransform(
        grid_name_to_nk_lookup=gridname_to_nkl,
        grid_name_to_wh=gname_to_wh,
        kb_tokenizer=kb_tokenizer,
        word_tokenizer=char_tokenizer,
        include_time=False,
        include_velocities=True,
        include_accelerations=True,
        kb_tokens_dtype=torch.int32,
        word_tokens_dtype=torch.int64
    )

    return full_transform



def get_traj_feats_and_distances_transform(grid_name: str, 
                                           grid: dict,
                                           char_tokenizer: CharLevelTokenizerv2,
                                           kb_tokenizer: KeyboardTokenizerv1,
                                           weights_func: Callable):
    assert isinstance(kb_tokenizer.i2t, list)
    grid_name_to_dist_lookup = {
        # Extra token is for legacy reasons
        grid_name: DistancesLookup(grid, kb_tokenizer.i2t + ['<extra_token>'])
    }

    full_transform = TrajFeats_KbWeights_FullTransform(
        grid_name_to_grid={grid_name: grid},
        grid_name_to_dist_lookup=grid_name_to_dist_lookup,
        word_tokenizer=char_tokenizer,
        include_time=False,
        include_velocities=True,
        include_accelerations=True,
        weights_func=weights_func,
        word_tokens_dtype=torch.int64,
    )

    return full_transform



def get_transforms(gridname_to_grid_path: str,
                   grid_name: str,
                   transform_name: str,
                   char_tokenizer: KeyboardTokenizerv1,
                   uniform_noise_range: int = 0,
                   dist_weights_func: Optional[Callable] = None,
                   ds_paths_list: Optional[List[str]] = None,
                   totals: Tuple[Optional[int], Optional[int]] = (None, None)
                   ) -> Tuple[Callable, Callable]:
    """Returns train and validation transforms."""
    
    grid = get_grid(grid_name, gridname_to_grid_path)
    w, h = grid['width'], grid['height']
    gname_to_wh = {grid_name: (w, h)}
    kb_tokenizer = KeyboardTokenizerv1()
    
    if transform_name == "traj_feats_and_nearest_key":

        full_transform = get_traj_and_nearest_key_transform(
            grid_name, grid, char_tokenizer, kb_tokenizer, gname_to_wh,
            uniform_noise_range, ds_paths_list, totals)
                
        
    elif transform_name == "traj_feats_and_distances":

        assert dist_weights_func is not None, "dist_weights_func must be provided"

        full_transform = get_traj_feats_and_distances_transform(
            grid_name, grid, char_tokenizer, kb_tokenizer, dist_weights_func)

    else:
        raise ValueError(f"Unknown transform name: '{transform_name}'")
    

    val_transform = full_transform

    train_transform = None
    if uniform_noise_range != 0:
        augmentation_transform = RandIntToTrajTransform(-uniform_noise_range, uniform_noise_range + 1)
        train_transform = SequentialTransform([augmentation_transform, full_transform])
    else:
        train_transform = full_transform

    return train_transform, val_transform
                