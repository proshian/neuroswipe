import pickle
from typing import Dict, Tuple

import numpy as np


class NearestKeyLookup:
    def __init__(self, grid, keyboard_selection_set):
        self._keyboard_selection_set = keyboard_selection_set
        self.grid = grid
        self.coord_to_kb_label = self._create_coord_to_kb_label(grid)
    
    def is_allowed_label(self, label: str) -> bool:
        if self._keyboard_selection_set is None:
            return True
        return label in self._keyboard_selection_set

    def _get_kb_label(self, key: dict) -> str:
        if 'label' in key:
            return key['label']
        if 'action' in key:
            return key['action']
        raise ValueError("Key has no label or action property")

    def _get_key_center(self, hitbox: Dict[str, int]) -> Tuple[int, int]:
        x = hitbox['x'] + hitbox['w'] / 2
        y = hitbox['y'] + hitbox['h'] / 2
        return x, y
    
    def _get_kb_label_without_map(self, x, y, grid: dict) -> str:
        """
        Returns label of the nearest key on the keyboard without using a map.
         
        Iterates over all keys and calculates the
        distance to (x, y) to find the nearest one.
        """
        nearest_kb_label = None
        min_dist = float("inf")

        for key in grid['keys']:
            label = self._get_kb_label(key)
            
            if not self.is_allowed_label(label):
                continue

            key_x, key_y = self._get_key_center(key['hitbox'])
            dist = (x - key_x)**2 + (y - key_y)**2
            if dist < min_dist:
                min_dist = dist
                nearest_kb_label = label 
        return nearest_kb_label
    

    def _create_coord_to_kb_label(self, grid: dict) -> np.array: # dtype = object
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
                coord_to_kb_label[x, y] = self._get_kb_label_without_map(x, y, grid)

        return coord_to_kb_label
    
    def get_nearest_kb_label(self, x, y):
        """
        Given coords on a keyboard (x, y) and its grid_name returns the nearest keyboard key

        By default it uses an array assosiated with grid_name
        that stores the nearest key label for every possible coord pair.

        If coords are outside of the keyboard boarders finds
        the nearest key by iterating over all keys.
        """        
        if x < 0 or x >= self.grid['width'] or y < 0 or y >= self.grid['height']:
            return self._get_kb_label_without_map(x, y, self.grid)
        return self.coord_to_kb_label[x, y]
        
    def save_state(self, path: str) -> None:
        state = {
            'selectoin_set': self._keyboard_selection_set,
            'coord_to_kb_label': self.coord_to_kb_label,
            'grid': self.grid
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
    @classmethod
    def load_state(cls, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)

        obj._keyboard_selection_set = state['selectoin_set']
        obj.grid = state['grid']
        obj.coord_to_kb_label = state['coord_to_kb_label']

        return obj
