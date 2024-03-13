import pickle
from typing import Dict, Tuple, Iterable

import numpy as np


class NearestKeyLookup:
    """
    Given a keyboard grid and a list of nearest_key_candidates
    returns the nearest key label for a given (x, y) coordinate.
    """

    def __init__(self, 
                 grid: dict, 
                 nearest_key_candidates: Iterable[str]) -> None:
        self._nearest_key_candidates = nearest_key_candidates
        self.grid = grid
        self.coord_to_kb_label = self._create_coord_to_kb_label(grid)
    
    def __call__(self, x, y):
        return self.get_nearest_kb_label(x, y)
    
    def is_allowed_label(self, label: str) -> bool:
        if self._nearest_key_candidates is None:
            return True
        return label in self._nearest_key_candidates

    def _get_kb_label(self, key: dict) -> str:
        if 'label' in key:
            return key['label']
        if 'action' in key:
            return key['action']
        raise ValueError("Key has no label or action property")

    def _get_key_center(self, hitbox: Dict[str, int]) -> Tuple[float, float]:
        x = hitbox['x'] + hitbox['w'] / 2
        y = hitbox['y'] + hitbox['h'] / 2
        return x, y
    
    def _get_kb_label_without_map(self, x, y) -> str:
        """
        Returns label of the nearest key on the keyboard without using a map.
         
        Iterates over all keys and calculates the
        distance to (x, y) to find the nearest one.
        """
        min_dist = float("inf")

        for key in self.grid['keys']:
            label = self._get_kb_label(key)
            
            if not self.is_allowed_label(label):
                continue

            key_x, key_y = self._get_key_center(key['hitbox'])
            dist = (x - key_x)**2 + (y - key_y)**2
            if dist < min_dist:
                min_dist = dist
                nearest_kb_label = label 
        return nearest_kb_label
    
    def _create_coord_to_kb_label(self, grid: dict) -> np.ndarray: # dtype = object
        coord_to_kb_label = np.zeros(
            (grid['width'], grid['height']), dtype=object)  # 1080 x 640 in our case
        coord_to_kb_label.fill('')

        for key in grid['keys']:
            label = self._get_kb_label(key)

            if not self.is_allowed_label(label):
                continue

            x_left = key['hitbox']['x']
            x_right = x_left + key['hitbox']['w']
            y_top = key['hitbox']['y']
            y_bottom = y_top + key['hitbox']['h']

            coord_to_kb_label[x_left:x_right, y_top:y_bottom] = label

        for x in range(grid['width']):
            for y in range(grid['height']):
                if coord_to_kb_label[x, y] != '':
                    continue
                coord_to_kb_label[x, y] = self._get_kb_label_without_map(x, y)

        return coord_to_kb_label
    
    def get_nearest_kb_label(self, x, y):
        """
        Returns the nearest key label for a given (x, y) coordinate.
        
        By default it uses an array assosiated that stores 
        the nearest key label for every coord pair within the keyboard.
        If the coordinate is out of bounds, finds the nearest key 
        by iterating over all keys (among nearest_key_candidates).
        """        
        if x < 0 or x >= self.grid['width'] or y < 0 or y >= self.grid['height']:
            return self._get_kb_label_without_map(x, y)
        return self.coord_to_kb_label[x, y]
    
    def _get_state(self) -> dict:
        state = {
            'nearest_key_candidates': self._nearest_key_candidates,
            'coord_to_kb_label': self.coord_to_kb_label,
            'grid': self.grid
        }
        return state
        
    def save_state(self, path: str) -> None:
        state = self._get_state()
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
    @classmethod
    def load_state(cls, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)

        obj._nearest_key_candidates = state['nearest_key_candidates']
        obj.grid = state['grid']
        obj.coord_to_kb_label = state['coord_to_kb_label']

        return obj


class ExtendedNearestKeyLookup(NearestKeyLookup):
    """
    In addition to the NearestKeyLookup, this class also stores
    the nearest key labels for the coordinates given in the extended_coords
    list. 
    This is useful during training when all the out-of-bounds
    coordinates are known and we can precompute the nearest key labels. 
    """
    def __init__(self, 
                 grid: dict, 
                 nearest_key_candidates: Iterable[str],
                 extended_coords: Iterable[Tuple[int, int]]) -> None:
        self._nearest_key_candidates = nearest_key_candidates
        self.grid = grid
        self.coord_to_kb_label = self._create_coord_to_kb_label(grid)
        self.extended_coord_to_kb_label = {
            (x,y): self._get_kb_label_without_map(x, y) 
            for x, y in extended_coords}

    def get_nearest_kb_label(self, x, y):
        if (x, y) in self.extended_coord_to_kb_label:
            return self.extended_coord_to_kb_label[(x, y)]
        return super().get_nearest_kb_label(x, y)
    
    def _get_state(self) -> dict:
        state = super()._get_state()
        state['extended_coord_to_kb_label'] = self.extended_coord_to_kb_label
        return state
    
    @classmethod
    def load_state(cls, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)

        obj._nearest_key_candidates = state['nearest_key_candidates']
        obj.grid = state['grid']
        obj.coord_to_kb_label = state['coord_to_kb_label']
        obj.extended_coord_to_kb_label = state['extended_coord_to_kb_label']

        return obj
    