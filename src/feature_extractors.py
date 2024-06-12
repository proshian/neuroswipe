"""
The dataset by default returns a tuple of 5 elements:
(X: array, Y: array, T: array, grid_name: str, tgt_word: str)

Feature extractors defined in this module are used 
to convert this tuple a tuple (model_in, model_out)
"""

from typing import Tuple, Dict, Optional, Iterable, List, Callable, Union, Set
from array import array
import json

import torch
from torch import Tensor
import numpy as np
from tqdm.auto import tqdm

from nearest_key_lookup import NearestKeyLookup, ExtendedNearestKeyLookup
from distances_lookup import DistancesLookup
from ns_tokenizers import KeyboardTokenizerv1, CharLevelTokenizerv2
from ns_tokenizers import ALL_CYRILLIC_LETTERS_ALPHABET_ORD
from dataset import RawDatasetEl 
from grid_processing_utils import get_gname_to_wh, get_kb_label, get_grid


DEFAULT_ALLOWED_KEYS = ALL_CYRILLIC_LETTERS_ALPHABET_ORD
GetItemTransformInput = Tuple[array, array, array, str, Optional[str], array]
EncoderInType = Union[Tensor, Tuple[Tensor, Tensor]]
FullTransformResultType = Tuple[Tuple[EncoderInType, Tensor], Tensor]


class FullTransform:
    def __init__(self, 
                 encoder_in_getter: Callable,
                 decoder_in_out_getter: Optional[Callable] = None
                 ) -> None:
        self.get_encoder_feats = encoder_in_getter
        self.get_decoder_in_out = decoder_in_out_getter
    
    def __call__(self, data: RawDatasetEl
                 ) -> FullTransformResultType:
        X, Y, T, grid_name, tgt_word = data
        encoder_in = self.get_encoder_feats(X, Y, T, grid_name)

        decoder_in, decoder_out = None, None
        if tgt_word is not None:
            assert self.get_decoder_in_out is not None, \
                "Decoder in/out getter is not provided, but tgt_word is not None."
            decoder_in, decoder_out = self.get_decoder_in_out(tgt_word)
        return (encoder_in, decoder_in), decoder_out


#################################################################################
####################  Helper functions and Callable classes  ####################
#################################################################################

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
    

class NearestKbTokensGetter:
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





####################   Related to weights aquiring   ####################

class DistancesGetter:
    def __init__(self, 
                 grid_name_to_dists_lookup: Dict[str, DistancesLookup],
                 dtype: torch.dtype = torch.float32
                 ) -> None:
        self.grid_name_to_dists_lookup = grid_name_to_dists_lookup
        self.dtype = dtype
    
    def _get_distances(self, X: Iterable, Y: Iterable, grid_name: str) -> Tensor:
        dl_lookup = self.grid_name_to_dists_lookup[grid_name]
        # distances = dl_lookup.get_distances_for_full_swipe_without_map(X, Y)
        distances = dl_lookup.get_distances_for_full_swipe_using_map(X, Y)
        distances = torch.tensor(distances, dtype=self.dtype)
        return distances


    def __call__(self, X: Iterable, Y: Iterable, T: Iterable, grid_name: str
                 ) -> Tensor:
        return self._get_distances(X, Y, grid_name)



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
        distances = self.distances_getter._get_distances(X, Y, grid_name)
        mask = (distances < 0)
        distances.masked_fill_(mask=mask, value = float('inf'))
        half_key_diag = self.grid_name_to_half_key_diag[grid_name]
        distances_scaled = distances / half_key_diag
        weights = self.weights_function(distances_scaled)
        weights.masked_fill_(mask=mask, value=0)
        return weights



########################### Weights functions ###########################

def weights_function_v1(distances: Tensor, bias = 4, scale = 1.8) -> Tensor:
    """
    Arguments:
    ----------
    distances: Tensor
        A 2d tensor where i-th element is a vector of distances 
        for i-th swipe dot and each of keyboard keys. It's supposed
        that distances are measured in half_key_diagonals 
        (distances = raw_distances / half_key_diagonal)

    $$f(x) = \frac{1}{1+e^{\frac{s \cdot x}{key\_radius} - b}}$$
    b = bias = 4
    s = scale = 1.8
    """
    
    # return 1 / (1 + torch.exp(1.8 * distances - 4))

    #! It may be a good idea to move division by half_key_diag outside
    # this function.  The division is just a scaling of distances
    # so that they are not in pixels but use half_key_diag as a unit. 
    sigmoid_input = distances * (-scale) + bias
    return torch.nn.functional.sigmoid(sigmoid_input)


def weights_function_v1_softmax(distances: Tensor, bias = 4, scale = 1.8) -> Tensor:
    """
    Arguments:
    ----------
    distances: Tensor
        A 2d tensor where i-th element is a vector of distances 
        for i-th swipe dot and each of keyboard keys. It's supposed
        that distances are measured in half_key_diagonals 
        (distances = raw_distances / half_key_diagonal)
    """
    mask = torch.isinf(distances)
    sigmoid_input = distances * (-scale) + bias
    weights = torch.nn.functional.sigmoid(sigmoid_input)
    # -inf to zero out unpresent values and have a sum of one 
    weights.masked_fill_(mask, float('-inf'))
    return torch.nn.functional.softmax(weights, dim=1)



def weights_function_sigmoid_normalized_v1(distances: Tensor, 
                                           bias = 4, scale = 1.8) -> Tensor:
    """
    Arguments:
    ----------
    distances: Tensor
        A 2d tensor where i-th element is a vector of distances 
        for i-th swipe dot and each of keyboard keys. It's supposed
        that distances are measured in half_key_diagonals 
        (distances = raw_distances / half_key_diagonal)
    
    $$f(x) = \frac{1}{1+e^{\frac{s \cdot x}{key\_radius} - b}}$$
    b = bias = 4
    s = scale = 1.8
    """
    #! It may be a good idea to move division by half_key_diag outside
    # this function.  The division is just a scaling of distances
    # so that they are not in pixels but use half_key_diag as a unit. 
    sigmoid_input = distances * (-scale) + bias
    sigmoidal_weights = torch.nn.functional.sigmoid(sigmoid_input)
    weights = sigmoidal_weights / sigmoidal_weights.sum(dim=1, keepdim=True)
    return weights


###################    util transforms    ###################

class SequentialTransform:
    def __init__(self, transforms) -> None:
        self.transforms = transforms
    
    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class TokensTypeCastTransform:
    def __call__(self, data: FullTransformResultType) -> FullTransformResultType:
        (traj_feats, kb_tokens, decoder_in), decoder_out = data
        # Embedding layer accepts int32, but not smaller types
        kb_tokens = kb_tokens.to(torch.int32)
        decoder_in = decoder_in.to(torch.int32)
        # CELoss accepts int64 only
        decoder_out = decoder_out.to(torch.int64)
        return (traj_feats, kb_tokens, decoder_in), decoder_out



#########################################################################
########################      Augmentations     #########################
#########################################################################

class RandIntToTrajTransform:
    def __init__(self, min_ = -3, max_ = 3) -> None:
        self.min = min_
        self.max = max_
        
    def __call__(self, data):
        X, Y, T, grid_name, tgt_word = data
        X = np.array(X, dtype = int) + np.random.randint(self.min, self.max, (len(X),))
        Y = np.array(Y, dtype = int) + np.random.randint(self.min, self.max, (len(Y),))
        return X, Y, T, grid_name, tgt_word


#########################################################################
########################  EncoderFeaturesGetters ########################
#########################################################################

class EncoderFeaturesGetter:
    def __call__(self, X: array, Y: array, T: array, grid_name: str) -> EncoderInType:
        raise NotImplementedError("EncoderFeaturesGetter is an abstract class.")


class EncoderFeaturesGetter_NearestKbTokens(EncoderFeaturesGetter):
    def __init__(self, 
                 grid_name_to_nk_lookup: Dict[str, NearestKeyLookup],
                 kb_tokenizer: KeyboardTokenizerv1,
                 dtype: torch.dtype = torch.int32,
                 input_to_int: bool = True
                 ) -> None:
        self.get_kb_tokens = NearestKbTokensGetter(
            grid_name_to_nk_lookup, kb_tokenizer, 
            return_tensor = True, dtype=dtype, input_to_int=input_to_int)
    
    def __call__(self, X: array, Y: array, T: array, grid_name: str) -> Tensor:
        return self.get_kb_tokens(X, Y, grid_name)



class EncoderFeaturesGetter_NearestKbTokensAndTrajFeats(EncoderFeaturesGetter):
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
        
        self._get_kb_tokens = NearestKbTokensGetter(
            grid_name_to_nk_lookup, kb_tokenizer, 
            return_tensor=True, dtype=kb_tokens_dtype, input_to_int=True)

    def __call__(self, X: array, Y: array, T: array, grid_name: str) -> Tuple[Tensor, Tensor]:
        # kb_tokens aquiring should be done first because
        # conversion to tensor would lead to indexing error since 
        # tensor(`index_val`) is not a proper index 
        # even if `index_val` is an integer.
        kb_tokens = self._get_kb_tokens(X, Y, grid_name)
        X, Y, T = (torch.tensor(arr, dtype=torch.float32) for arr in (X, Y, T))
        traj_feats = self._get_traj_feats(X, Y, T, grid_name)
        
        return traj_feats, kb_tokens


class EncoderFeaturesGetter_KbKeyWeightsAndTrajFeats(EncoderFeaturesGetter):
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
        # weights aquiring should be done first because
        # conversion to tensor would lead to indexing error since 
        # tensor(`index_val`) is not a proper index 
        # even if `index_val` is an integer.
        weights = self.get_weights(X, Y, grid_name)

        X, Y, T = (torch.tensor(arr, dtype=torch.float32) for arr in (X, Y, T))
        traj_feats = self._get_traj_feats(X, Y, T, grid_name)
        
        return traj_feats, weights




class EncoderFeaturesTupleGetter:
    """
    Given a tuple of two feature getters: 
        1) that extracts features as points of a trajectory
        2) that extracts features from point location on keyboard
    
    Returns a tuple of two tensors:
        1) Trajectory point features
        2) Keyboard point features
    """
    def __init__(self, traj_feats_getter: Callable, kb_feats_getter: Callable, 
                 kb_uses_t: bool = True) -> None:
        self.traj_feats_getter = traj_feats_getter
        self.kb_feats_getter = kb_feats_getter
        self.kb_uses_t = kb_uses_t

    def __call__(self, X: array, Y: array, T: array, grid_name: str) -> Tuple[Tensor, Tensor]:
        traj_feats = self.traj_feats_getter(X, Y, T, grid_name)
        if self.kb_uses_t:
            kb_feats = self.kb_feats_getter(X, Y, T, grid_name)
        else:
            kb_feats = self.kb_feats_getter(X, Y, grid_name)
        return traj_feats, kb_feats




class EncoderFeaturesGetter_KbKeyDistancesAndTrajFeats(EncoderFeaturesTupleGetter):
    def __init__(self, 
                 distances_getter: Callable,
                 grid_name_to_grid: Dict[str, dict],
                 include_time: bool,
                 include_velocities: bool,
                 include_accelerations: bool) -> None:
        gname_to_wh = get_gname_to_wh(grid_name_to_grid)
        traj_feats_getter = TrajFeatsGetter(
            gname_to_wh,
            include_time, include_velocities, include_accelerations)
    
        super().__init__(traj_feats_getter, distances_getter, kb_uses_t=True)

                 

#########################################################################
########################  DecoderInputOutputGetter  #####################
#########################################################################

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




######################################################################
######################################################################
###################    Full transforms aquiring    ###################
######################################################################
######################################################################


def get_grid(grid_name: str, grids_path: str) -> dict:
    with open(grids_path, "r", encoding="utf-8") as f:
        return json.load(f)[grid_name]



def get_gridname_to_out_of_bounds_coords_dict(
        data_paths: List[str], gridname_to_wh: dict,
        totals: Iterable[Optional[int]] = None
        ) -> Dict[str, Set[Tuple[int, int]]]:
    """
    Returns a dictionary with grid names as keys and
    lists of out of bounds coordinates present in the dataset as values.
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
        


def get_extra_coords_dict(ds_paths_list: List[str], 
                          gridname_to_wh: Dict[str, Tuple[int, int]],
                          uniform_noise_range: int = 0,
                          totals: Optional[List[int]] = None
                          ) -> Dict[str, Set[Tuple[int, int]]]:
    print("Accumulating out-of-bounds coordinates...")
    gname_to_out_of_bounds = get_gridname_to_out_of_bounds_coords_dict(
        ds_paths_list, 
        gridname_to_wh = gridname_to_wh,
        totals=totals
    )

    print("augmenting gname_to_out_of_bounds")
    gname_to_out_of_bounds = update_out_of_bounds_with_noise(
        noise_min = -uniform_noise_range, noise_max=uniform_noise_range+1,
        gname_to_out_of_bounds = gname_to_out_of_bounds, gridname_to_wh = gridname_to_wh,
    )
    
    return gname_to_out_of_bounds



def get_gname_to_nkl(gname_to_grid: Dict[str, dict],
                     gname_to_out_of_bounds: Dict[str, Set[Tuple[int, int]]]
                        ) -> Dict[str, ExtendedNearestKeyLookup]:
    
    gridname_to_nkl = {
        gname: ExtendedNearestKeyLookup(
            grid, ALL_CYRILLIC_LETTERS_ALPHABET_ORD,
            gname_to_out_of_bounds[gname]
        ) for gname, grid in gname_to_grid.items()
    }

    return gridname_to_nkl



def get_traj_feats_and_weights_transform(gname_to_grid: Dict[str, dict],
                                           char_tokenizer: CharLevelTokenizerv2,
                                           grid_name_to_dists_lookup: Dict[str, DistancesLookup],
                                           weights_func: Callable,
                                           include_time: bool,
                                           include_velocities: bool,
                                           include_accelerations: bool
                                            ) -> Callable:
    full_transform = FullTransform(
        encoder_in_getter=EncoderFeaturesGetter_KbKeyWeightsAndTrajFeats(
            grid_name_to_dists_lookup, gname_to_grid,
            include_time=include_time, include_velocities=include_velocities,
            include_accelerations=include_accelerations, weights_func=weights_func
        ),
        decoder_in_out_getter=DecoderInputOutputGetter(
            word_tokenizer=char_tokenizer,
            dtype=torch.int64
        )
    )

    return full_transform


def assert_traj_feats_provided(include_time: Optional[bool],
                               include_velocities: Optional[bool],
                               include_accelerations: Optional[bool]) -> None:
    assert include_time is not None, "include_time must be provided"
    assert include_velocities is not None, "include_velocities must be provided"
    assert include_accelerations is not None, "include_accelerations must be provided"



def get_val_transform(gridname_to_grid_path: str,
                      grid_names: List[str],
                      transform_name: str,
                      char_tokenizer: KeyboardTokenizerv1,
                      uniform_noise_range: int = 0,
                      include_time: Optional[bool] = None,
                      include_velocities: Optional[bool] = None,
                      include_accelerations: Optional[bool] = None,
                      dist_weights_func: Optional[Callable] = None,
                      ds_paths_list: Optional[List[str]] = None,
                      totals: Tuple[Optional[int], Optional[int]] = None
                   ) -> Tuple[Callable, Callable]:
    """Returns validation transform"""
    TRAJ_FEATS_AND_WEIGHTS = "traj_feats_and_distances"  # Not a mistake; Legacy name
    TRAJ_FEATS_AND_DISTANCES = "traj_feats_and_distances__actual"
    TRAJ_FEATS_AND_NEAREST_KEY = "traj_feats_and_nearest_key"
    NEAREST_KEY_ONLY = "nearest_key_only"


    transforms_need_out_of_bounds = [TRAJ_FEATS_AND_NEAREST_KEY, NEAREST_KEY_ONLY]
    transforms_need_nkl = [TRAJ_FEATS_AND_NEAREST_KEY, NEAREST_KEY_ONLY]
    transforms_need_dist_lookup = [TRAJ_FEATS_AND_WEIGHTS, TRAJ_FEATS_AND_DISTANCES]
    
    gname_to_grid = {gname: get_grid(gname, gridname_to_grid_path) 
                     for gname in grid_names}

    gname_to_wh = {gname: (grid['width'], grid['height']) for gname, grid in gname_to_grid.items()}

    kb_tokenizer = KeyboardTokenizerv1()


    if transform_name in transforms_need_out_of_bounds:
        assert ds_paths_list is not None, "ds_paths_list must be provided"
        if ds_paths_list is None:
            print("Warning: no ds_paths_list provided. gname_to_out_of_bounds will be empty.")
            gname_to_out_of_bounds = {gname: set() for gname in gname_to_wh.keys()}
        else:
            gname_to_out_of_bounds = get_extra_coords_dict(
                ds_paths_list, gname_to_wh, uniform_noise_range, totals
            )
    
    if transform_name in transforms_need_nkl:
        gridname_to_nkl = get_gname_to_nkl(gname_to_grid, gname_to_out_of_bounds)

    if transform_name in transforms_need_dist_lookup:
        assert isinstance(kb_tokenizer.i2t, list)
        grid_name_to_dist_lookup = {
            # Extra token is for legacy reasons
            gname: DistancesLookup(grid, kb_tokenizer.i2t + ['<extra_token>'])
            for gname, grid in gname_to_grid.items()
        }
        gridname_to_dists_lookup = {gname: DistancesLookup(grid) for gname, grid in gname_to_grid.items()}

     

    if transform_name == TRAJ_FEATS_AND_NEAREST_KEY:
        assert_traj_feats_provided(include_time, include_velocities, include_accelerations)

        full_transform = FullTransform(
                encoder_in_getter=EncoderFeaturesGetter_NearestKbTokensAndTrajFeats(
                grid_name_to_nk_lookup=gridname_to_nkl,
                grid_name_to_wh=gname_to_wh,
                kb_tokenizer=kb_tokenizer,
                include_time=include_time,
                include_velocities=include_velocities,
                include_accelerations=include_accelerations
            ),
            decoder_in_out_getter=DecoderInputOutputGetter(
                word_tokenizer=char_tokenizer,
                dtype=torch.int64
            )
        )

                
    elif transform_name == TRAJ_FEATS_AND_WEIGHTS:
        assert_traj_feats_provided(include_time, include_velocities, include_accelerations)
        assert dist_weights_func is not None, "dist_weights_func must be provided"

        full_transform = get_traj_feats_and_weights_transform(
            gname_to_grid, char_tokenizer, gridname_to_dists_lookup, dist_weights_func,
            include_time, include_velocities, include_accelerations
        )

    elif transform_name == NEAREST_KEY_ONLY:
        full_transform = FullTransform(
            encoder_in_getter=EncoderFeaturesGetter_NearestKbTokens(
                grid_name_to_nk_lookup=gridname_to_nkl,
                kb_tokenizer=kb_tokenizer,
                dtype=torch.int32
            ),
            decoder_in_out_getter=DecoderInputOutputGetter(
                word_tokenizer=char_tokenizer,
                dtype=torch.int64
            )
        )
    
    elif transform_name == TRAJ_FEATS_AND_DISTANCES:
        assert_traj_feats_provided(include_time, include_velocities, include_accelerations)
        
        full_transform = FullTransform(
            encoder_in_getter=EncoderFeaturesTupleGetter(
                traj_feats_getter=TrajFeatsGetter(gname_to_wh, include_time, include_velocities, include_accelerations),
                kb_feats_getter=DistancesGetter(grid_name_to_dist_lookup, dtype=torch.float32),
                kb_uses_t=True
            ),
            decoder_in_out_getter=DecoderInputOutputGetter(
                word_tokenizer=char_tokenizer,
                dtype=torch.int64
            )
        )

    else:
        raise ValueError(f"Unknown transform name: '{transform_name}'")
    
    return full_transform



def get_transforms(gridname_to_grid_path: str,
                     grid_names: List[str],
                     transform_name: str,
                     char_tokenizer: KeyboardTokenizerv1,
                     uniform_noise_range: int = 0,
                     include_time: Optional[bool] = None,
                     include_velocities: Optional[bool] = None,
                     include_accelerations: Optional[bool] = None,
                     dist_weights_func: Optional[Callable] = None,
                     ds_paths_list: Optional[List[str]] = None,
                     totals: Tuple[Optional[int], Optional[int]] = None
                     ) -> Tuple[Callable, Callable]:
    """Returns train and validation transforms"""
    
    
    val_transform = get_val_transform(
        gridname_to_grid_path, grid_names, transform_name, char_tokenizer,
        uniform_noise_range, include_time, include_velocities,
        include_accelerations, dist_weights_func, ds_paths_list, totals
    )

    train_transform = val_transform
    if uniform_noise_range != 0:
        augmentation_transform = RandIntToTrajTransform(-uniform_noise_range, uniform_noise_range + 1)
        train_transform = SequentialTransform([augmentation_transform, val_transform])

    return train_transform, val_transform
                