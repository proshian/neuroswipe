from typing import Optional, List, Dict, Tuple, Any
import pickle

import numpy as np



def distance(dots: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Arguments:
    ----------
    dots: np.ndarray
        Dots tensor. dots.shape = (*DOT_DIMS, 2). DOT_DIMS: tuple = (S1, S2, S3, ... SD).
    centers: np.ndarray
        Centers tensor. centers.shape = (K, 2). K is number of centers.

    Returns:
    --------
    np.ndarray
        Distance tensor. Distance tensor.shape = (*DOT_DIMS, K).
        Squared euclidean distance is used.
    
    Example:
    --------
    dots = np.array([[1, 2], [3, 4], [5, 6]])
    centers = np.array([[1, 2], [3, 4]])
    distance(dots, centers) -> np.array([[0, 8], [8, 0], [32, 8]])
    """
    # (K, 2) -> (1, 1, ..., 1, K, 2), 
    centers = centers.reshape([1]*(dots.ndim - 1) + list(centers.shape))
    dots = np.expand_dims(dots, -2)  # (*DOT_DIMS, 1, 2)
    return np.power((centers - dots), 2).sum(axis=-1)


class DistancesLookup:
    """
    Given a coordinate (x,y) returns the distance to all keys.
    """

    def __init__(self, grid: dict, kb_key_list: Optional[List[str]] = None, 
                 return_dict: bool = False, 
                 raise_on_key_not_in_grid: bool = False,
                 fill_unpresent_centers_val: float = -1,
                 fill_unpresent_dist_val: float = -1) -> None:
        """
        Arguments:
        ----------
        grid: dict
        kb_key_list: Optional[List[str]]
            An ordered list of keys. If None, all keys from the grid are used.
            Distances will be returned in the same order as in kb_key_list.
        return_dict: bool
            If True, the distances will be returned as a dict.
        raise_on_key_not_in_grid: bool
            If True, raises an error if a key from kb_key_list 
            is not present in the grid.
        fill_unpresent_centers_val: float
            Value to fill for centers of keys that are not present in the grid.
            Defaults to -1 since all centers have positive coordinates.
        fill_unpresent_dist_val: float
            Value to fill for distances to keys that are not present in the grid.
            Defaults to -1 because it's easy to spot since all distances are positive.
        """
        self.grid = grid
        self.return_dict = return_dict
        self.KB_KEY_LIST = kb_key_list or self._get_all_key_labels()
        self.i_to_kb_key = self.KB_KEY_LIST
        self.kb_key_to_i = {kb_key: i for i, kb_key in enumerate(self.KB_KEY_LIST)}
        self.fill_unpresent_centers_val = fill_unpresent_centers_val
        self.fill_unpresent_dist_val = fill_unpresent_dist_val

        if raise_on_key_not_in_grid:
            self._check_all_keys_in_grid()

        self.centers = self._get_centers()

        self.coord_to_distances = self._create_coord_to_distances()

    def __call__(self, x, y):
        return self.get_distances(x, y)
    

    def mask_unpresent_distance(self, distances: np.ndarray) -> np.ndarray:
        """
        Given distances of shape (*DOT_DIMS, K) where K is the number of keys,
        and DOT_DIMS is a tuple of arbitrary length,
        returns a tensor where last dim is masked with fill_unpresent_dist_val.
        """
        mask = self.centers[:, 0] == self.fill_unpresent_centers_val
        distances[..., mask] = self.fill_unpresent_dist_val
        return distances
    


    @staticmethod
    def _distance_raw(dots: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return distance(dots, centers)
    
    def _distance(self, dots: np.ndarray, centers: np.ndarray) -> np.ndarray:
        dist = self._distance_raw(dots, centers)
        return self.mask_unpresent_distance(dist)
    
    def get_distances_for_full_swipe_without_map(self, X: list, Y: list) -> np.ndarray:
        """
        Returns the distances for a full swipe.
        """
        dots = np.array([X, Y]).T
        swipe_distances = self._distance(dots, self.centers)
        return swipe_distances
    
    def get_distances_for_full_swipe_using_map(self, X: list, Y: list) -> np.ndarray:
        """
        Returns the distances for a full swipe.
        """
        swipe_distances = np.zeros((len(X), len(self.centers)))
        for i, (x, y) in enumerate(zip(X, Y)):
            swipe_distances[i, :] = self.get_distances(x, y)
        return self.mask_unpresent_distance(swipe_distances)
    
    def _check_all_keys_in_grid(self) -> None:
        all_grid_kb_laybels = set(self._get_all_key_labels()) 
        for kb_key in self.KB_KEY_LIST:
            if kb_key not in all_grid_kb_laybels:
                raise ValueError(f"Key {kb_key} is not present in the grid")

    def _get_kb_label(self, key: dict) -> str:
        if 'label' in key:
            return key['label']
        if 'action' in key:
            return key['action']
        raise ValueError("Key has no label or action property")
    
    def _get_all_key_labels(self) -> List[str]:
        return [self._get_kb_label(key) for key in self.grid['keys']]

    def _get_key_center(self, hitbox: Dict[str, int]) -> Tuple[float, float]:
        x = hitbox['x'] + hitbox['w'] / 2
        y = hitbox['y'] + hitbox['h'] / 2
        return x, y
        
    def _get_centers(self) -> np.ndarray:
        centers = np.full(
            (len(self.i_to_kb_key), 2), 
            self.fill_unpresent_centers_val)
        
        for key in self.grid['keys']:
            label = self._get_kb_label(key)
            if label in self.kb_key_to_i:
                centers[self.kb_key_to_i[label]] = self._get_key_center(key['hitbox'])
        return centers
        
    def _create_coord_to_distances(self) -> np.ndarray:
        w, h = self.grid['width'], self.grid['height']
        dots = np.indices((w, h)).transpose(1, 2, 0)  # (w, h, 2)
        return self._distance(dots, self.centers)

    def get_distances_arr(self, x: int, y: int) -> np.ndarray:
        if x < 0 or x >= self.grid['width'] or y < 0 or y >= self.grid['height']:
            return self._distance(np.array([[x, y]]), self.centers).flatten()
        return self.coord_to_distances[x, y]

    def get_distances(self, x: int, y: int) -> Dict[str, float] | np.ndarray:
        dist_arr = self.get_distances_arr(x, y)
        if self.return_dict:
            return {kb_key: dist for kb_key, dist in zip(self.KB_KEY_LIST, dist_arr)}
        return dist_arr
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            'grid': self.grid,
            'kb_key_list': self.KB_KEY_LIST,
            'return_dict': self.return_dict,
            'coord_to_distances': self.coord_to_distances,
            'centers': self.centers,
            'fill_unpresent_dist_val': self.fill_unpresent_dist_val,
            'fill_unpresent_centers_val': self.fill_unpresent_centers_val
        }

    def save_state(self, path: str) -> None:
        state = self._get_state()
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load_state(cls, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.grid = state['grid']
        obj.KB_KEY_LIST = state['kb_key_list']
        obj.return_dict = state['return_dict']
        obj.coord_to_distances = state['coord_to_distances']
        obj.centers = state['centers']
        obj.fill_unpresent_dist_val = state['fill_unpresent_dist_val']
        obj.fill_unpresent_centers_val = state['fill_unpresent_centers_val']
        obj.i_to_kb_key = obj.KB_KEY_LIST
        obj.kb_key_to_i = {kb_key: i for i, kb_key in enumerate(obj.KB_KEY_LIST)}

        return obj
    