import json
from typing import Optional, List, Tuple, Dict
import array

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from tokenizers import CharLevelTokenizerv2, KeyboardTokenizerv1


class NeuroSwipeDatasetv2(Dataset):
    """
    Dataset class for NeuroSwipe dataset.
    The dataset file weights over 3 GB and contains over 6 million swipe gestures.
    """

    def __init__(self,
                 data_path: str,
                 gridname_to_grid: dict,
                 kb_tokenizer: KeyboardTokenizerv1,
                 max_traj_len: int,
                 word_tokenizer: CharLevelTokenizerv2,  # should contain max word len
                 include_time: bool = False,
                 include_velocities: bool = True,
                 include_accelerations: bool = True,
                 include_grid_name: bool = False,
                 has_target = True,
                 has_one_grid_only = True,
                 keyboard_selection_set = None,
                 total: Optional[int] = None):
        """
        Arguments:
        -----------
        data_path (string): Path to the NeuroSwipe dataset in JSON format.
            A custom version of the dataset is used:
            "grid" property is replaced with "grid_name" property.
        """
        if include_accelerations and not include_velocities:
            raise ValueError("Accelerations are supposed \
                             to be an addition to velocities. Add velocities.")
        
        if has_one_grid_only and len(gridname_to_grid) != 1:
            raise ValueError(f"has_one_grid_only is True \
                             but len(gridname_to_grid) != 1")

        self.max_traj_len = max_traj_len
        self.include_velocities = include_velocities
        self.include_accelerations = include_accelerations
        self.include_time = include_time
        self.has_target = has_target
        self.include_grid_name = include_grid_name
        self.keyboard_selection_set = keyboard_selection_set

        self.word_tokenizer = word_tokenizer

        self._grid_name_to_grid = gridname_to_grid

        self._nearest_kb_label_dict = self._get_nearest_kb_label_dict(gridname_to_grid)

        self.data_list = []
        self._set_data(data_path, gridname_to_grid, kb_tokenizer, self.data_list, total = total)


    def get_nearest_kb_label(self, x, y, grid_name, gridname_to_grid):
        """
        Given coords on a keyboard (x, y) and its grid_name returns the nearest keyboard keys

        By default it uses an array assosiated with grid_name
        that stores the nearest key label for every possible coord pair.

        However sometimes the coords are outside of the keyboard boarders.
        In this case we find the nearest key in a loop.
        """        
        grid = gridname_to_grid[grid_name]
        if x < 0 or x >= grid['width'] or y < 0 or y >= grid['height']:
            return self._get_kb_label_without_map(x, y, grid)
        else:
            return self._nearest_kb_label_dict[grid_name][x, y]
    


    def _get_key_center(self, hitbox: Dict[str, int]) -> Tuple[int, int]:
        x = hitbox['x'] + hitbox['w'] / 2
        y = hitbox['y'] + hitbox['h'] / 2
        return x, y
    
    def _get_kb_label(self, key: dict) -> str:
        if 'label' in key:
            return key['label']
        if 'action' in key:
            return key['action']
        raise ValueError("Key has no label or action")


    def _get_kb_label_without_map(self, x, y, grid: dict) -> str:
        """
        Returns label of the nearest key on the keyboard without using a map.
         
        Iterates over all keys and calculates the distance to (x, y) to find the nearest one.
        """
        nearest_kb_label = None
        min_dist = float("inf")

        for key in grid['keys']:
            label = self._get_kb_label(key)
            
            if self.keyboard_selection_set and label not in self.keyboard_selection_set:
                continue

            key_x, key_y = self._get_key_center(key['hitbox'])
            dist = (x - key_x)**2 + (y - key_y)**2
            if dist < min_dist:
                min_dist = dist
                nearest_kb_label = label 
        return nearest_kb_label


    def _get_nearest_kb_label_dict(self, gridname_to_grid: dict) -> Dict[str, np.chararray]:
        """
        Creates a dict that maps grid_name to a map (np.chararray) from coordinate to nearest key label.
        """
        nearest_kb_label_dict = {}
        for grid_name, grid in gridname_to_grid.items():
            nearest_kb_label_dict[grid_name] = self._get_coord_to_kb_label(grid)
        return nearest_kb_label_dict
    

    def _get_coord_to_kb_label(self, grid: dict) -> np.array: # dtype = object
        coord_to_kb_label = np.zeros(
            (grid['width'], grid['height']), dtype=object)  # 1080 x 640 in our case
        coord_to_kb_label.fill('')

        for key in grid['keys']:
            label = self._get_kb_label(key)

            if self.keyboard_selection_set and label not in self.keyboard_selection_set:
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
            

    def _set_data(self,
                  data_path: str,
                  gridname_to_grid: dict,
                  kb_tokenizer,
                  data_list: list,
                  total: Optional[int] = None):
        with open(data_path, "r", encoding="utf-8") as json_file:
            for line in tqdm(json_file, total = total):
                data_list.append(self._get_data_from_json_line(line, gridname_to_grid, kb_tokenizer))


    def _get_dx_dt(self,
                   X: torch.tensor,
                   T: torch.tensor,
                   len: int) -> List[float]:
        """
        Calculates dx/dt for a list of x coordinates and a list of t coordinates.

        Arguments:
        ----------
        X : torch.tensor
            x (position) coordinates.
        T : torch.tensor
            T[i] = time from the beginning of the swipe corresponding to X[i].
        len : int
            Length of the swipe trajectory. Indexes greater than len are ignored.
        """
        dx_dt = torch.zeros_like(X)
        # dx_dt[1:-1] = (X[2:] - X[:-2]) / (T[2:] - T[:-2])
        dx_dt[1:len-1] = (X[2:len] - X[:len-2]) / (T[2:len] - T[:len-2])

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
    

    def _get_data_from_json_line(self,
                                 line,
                                 gridname_to_grid,
                                 kb_tokenizer) -> Tuple[list, list, list, str]:
        """
        Parses a JSON line and returns a dictionary with data.
        """
        data = json.loads(line)

        X = array.array('h', data['curve']['x'])
        Y = array.array('h', data['curve']['y'])
        T = array.array('h', data['curve']['t'])

        grid_name = data['curve']['grid_name']   

        kb_labels = [self.get_nearest_kb_label(x, y, grid_name, gridname_to_grid) for x, y in zip(X, Y)]
        kb_tokens = [kb_tokenizer.get_token(label) for label in kb_labels]
        kb_tokens += [kb_tokenizer.get_token('<pad>')] * (self.max_traj_len - len(kb_labels))
        kb_tokens = array.array('h', kb_tokens)

        if not self.has_target:
            return X, Y, T, kb_tokens, grid_name
        else:
            word: str = data['word']
            return X, Y, T, kb_tokens, word, grid_name


    def __len__(self):
        return len(self.data_list)
    

    def __getitem__(self, idx):
        if self.has_target:
            X_list, Y_list, T_list, kb_tokens, word, grid_name = self.data_list[idx]
        else:
            X_list, Y_list, T_list, kb_tokens, grid_name = self.data_list[idx]

        X = torch.zeros(self.max_traj_len, dtype=torch.float32)
        Y = torch.zeros(self.max_traj_len, dtype=torch.float32)
        T = torch.zeros(self.max_traj_len, dtype=torch.float32)

        X[:len(X_list)] = torch.tensor(X_list, dtype=torch.float32)
        Y[:len(Y_list)] = torch.tensor(Y_list, dtype=torch.float32)
        T[:len(T_list)] = torch.tensor(T_list, dtype=torch.float32)

        xyt = torch.cat(
            (
                X.reshape(-1, 1),
                Y.reshape(-1, 1),
            ),
            axis = 1
        )

        if self.include_time:
            xyt = torch.cat(
                (
                    xyt,
                    T.reshape(-1, 1)
                ),
                axis = 1
            )

        traj_len = len(X_list)

        if self.include_velocities:
            dx_dt = self._get_dx_dt(X, T, traj_len)
            dy_dt = self._get_dx_dt(Y, T, traj_len)
            xyt = torch.cat(
                [
                    xyt,
                    dx_dt.reshape(-1, 1),
                    dy_dt.reshape(-1, 1)
                ],
                axis = 1
            )

        if self.include_accelerations:
            d2x_dt2 = self._get_dx_dt(dx_dt, T, traj_len)
            d2y_dt2 = self._get_dx_dt(dy_dt, T, traj_len)
            xyt = torch.cat(
                [
                    xyt,
                    d2x_dt2.reshape(-1, 1),
                    d2y_dt2.reshape(-1, 1)
                ],
                axis = 1
            )

        traj_pad_mask = torch.ones(self.max_traj_len, dtype=torch.bool)
        traj_pad_mask[:len(X_list)] = False


        kb_tokens = torch.tensor(kb_tokens, dtype=torch.int64)

        decoder_out_char_seq = None
        decoder_in_char_seq = None
        word_mask = None

        if self.has_target:
            # <sos>, token1, token2, ... token_n, <eos>
            token_seq: List[int] = self.word_tokenizer.encode(word)
            token_seq = torch.tensor(token_seq, dtype = torch.int64)

            # model inputs and outputs are one token smaller than max_word,
            # Model inputs: <sos>, token1, ... token_n, <pad_0>, <pad_1>, ... <pad_k>
            # Model outputs:       token1, ... token_n, <EOS!>,  <pad_1>, ... <pad_k>
            decoder_seq_len = self.word_tokenizer.max_word_len - 1

            
            word_mask = torch.ones(decoder_seq_len, dtype=torch.bool)
           
            # <sos> and full word are not masked;
            # <eos> and all <pad> are masked.
            word_mask[:len(word) + 1] = False 
            
            # <sos>, token1, ... token_n
            decoder_in_char_seq = torch.full(
                (decoder_seq_len,),
                self.word_tokenizer.char_to_idx['<pad>'],
                dtype=torch.int64)
            decoder_in_char_seq[:len(word) + 1] = token_seq[:-1]

            # token1, ... token_n, <eos>
            decoder_out_char_seq = torch.full(
                (decoder_seq_len,),
                self.word_tokenizer.char_to_idx['<pad>'],
                dtype=torch.int64)
            decoder_out_char_seq[:len(word) + 1] = token_seq[1:]

        # ! MAybe will generate derivatives from X_list and Y_list.
        # Then, this section will be moved above
        grid = self._grid_name_to_grid[grid_name]
        xyt[:len(X_list), 0] = xyt[:len(X_list), 0] / grid['width'] 
        xyt[:len(Y_list), 1] = xyt[:len(X_list), 1] / grid['height']
        
        if self.include_grid_name:
            return (xyt, kb_tokens, decoder_in_char_seq, traj_pad_mask, word_mask), decoder_out_char_seq, grid_name
        
        return (xyt, kb_tokens, decoder_in_char_seq, traj_pad_mask, word_mask), decoder_out_char_seq





class NeuroSwipeGridSubset(Dataset):
    def __init__(self, dataset: Dataset, grid_name: str):
        self.dataset = dataset
        self.grid_name = grid_name
        self.grid_name_idxs = self._get_grid_name_idxs()
        
            
    def _get_grid_name_idxs(self):
        grid_name_idxs: list[int] = []
        for i, ((_, _, _, _, _), _, grid_name) in enumerate(self.dataset):
            if grid_name == self.grid_name:
                grid_name_idxs.append(i)
        return grid_name_idxs

    
    def __len__(self):
        return len(self.grid_name_idxs)
    
    def __getitem__(self, idx):
        return self.dataset[self.grid_name_idxs[idx]]