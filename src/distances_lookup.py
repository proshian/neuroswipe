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
    d =  np.power((centers - dots), 2).sum(axis=-1)
    return np.sqrt(d)

# For unit tests:

# N_KEYS = 30
# WIDTH = MAX_COORD = 1080
# HEIGHT = 667
# dots = np.indices((WIDTH, HEIGHT)).transpose(1, 2, 0)  # (WIDTH, HEIGHT, 2)
# centers = np.random.randint(0, MAX_COORD, (N_KEYS, 2))
# result = distance(dots, centers)
# assert result.shape == (WIDTH, HEIGHT, N_KEYS)

# dots = np.random.randint(0, MAX_COORD, (100, 50, 90, 2))
# centers = np.random.randint(0, MAX_COORD, (N_KEYS, 2))
# result = distance(dots, centers)
# assert result.shape == (100, 50, 90, N_KEYS)



class DistancesLookup:
    """
    Given a coordinate (x,y) returns the distance to all keys.
    """

    def __init__(self, grid: dict, kb_key_list: Optional[List[str]] = None, 
                 return_dict: bool = False, 
                 raise_on_key_not_in_grid: bool = False) -> None:
        """
        Given a keyboard grid and optionally a allowed_keys list,
        """
        self.grid = grid
        self.return_dict = return_dict
        self.KB_KEY_LIST = kb_key_list or self._get_all_key_labels(grid)
        self.i_to_kb_key = self.KB_KEY_LIST
        self.kb_key_to_i = {kb_key: i for i, kb_key in enumerate(self.KB_KEY_LIST)}
        self.centers = self._get_centers()

        self.coord_to_distances = self._create_coord_to_distances()

    def __call__(self, x, y):
        return self.get_distances(x, y)


    @staticmethod
    def _distance(dots: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return distance(dots, centers)
    
    def get_distances_for_full_swipe_v1(self, X: list, Y: list) -> np.ndarray:
        """
        Returns the distances for a full swipe.
        """
        dots = np.array([X, Y]).T
        return self._distance(dots, self.centers)
    
    def get_distances_for_full_swipe_v2(self, X: list, Y: list) -> List[np.ndarray]:
        """
        Returns the distances for a full swipe.
        """
        return [self.get_distances(x, y) for x, y in zip(X, Y)]
    
    def _get_kb_label(self, key: dict) -> str:
        if 'label' in key:
            return key['label']
        if 'action' in key:
            return key['action']
        raise ValueError("Key has no label or action property")
    
    def _get_all_key_labels(self, grid: dict) -> List[str]:
        return [self._get_kb_label(key) for key in grid['keys']]

    def _get_key_center(self, hitbox: Dict[str, int]) -> Tuple[float, float]:
        x = hitbox['x'] + hitbox['w'] / 2
        y = hitbox['y'] + hitbox['h'] / 2
        return x, y
        
    def _get_centers(self, fill_unpresent = np.nan) -> np.ndarray:
        centers = np.empty((len(self.i_to_kb_key), 2))
        centers.fill(fill_unpresent)
        
        for key in self.grid['keys']:
            label = self._get_kb_label(key)
            if label in self.kb_key_to_i:
                centers[self.kb_key_to_i[label]] = self._get_key_center(key['hitbox'])
        return centers
        
    def _create_coord_to_distances(self, w: int, h: int) -> np.ndarray:
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
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            'grid': self.grid,
            'kb_key_list': self.KB_KEY_LIST,
            'return_dict': self.return_dict,
            'coord_to_distances': self.coord_to_distances,
            'centers': self.centers,
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
        obj.i_to_kb_key = obj.KB_KEY_LIST
        obj.kb_key_to_i = {kb_key: i for i, kb_key in enumerate(obj.KB_KEY_LIST)}

        return obj