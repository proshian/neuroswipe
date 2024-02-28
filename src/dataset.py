import json
from typing import Optional, List, Tuple, Dict, Set
import array

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from tokenizers import CharLevelTokenizerv2, KeyboardTokenizerv1


class NeuroSwipeDatasetv3(Dataset):
    """
    Dataset class for NeuroSwipe dataset.

    The dataset uses all data from a given json in the same order as the json.
    There are separate json files for every grid.

    Given a NeuroSwipeDatasetv3 object nsd, nsd[i] is a tuple:
    ((trajectory_features, k_key_tokens, decoder_in_char_seq), decoder_out_char_seq)
    
    ! Warning: refactoring planned so that the output is a dictionary.

    WARNING:
    The class is in the process of refactoring. The padding will be done
    in DataLoaders collate_fn instead of __getitem__ method.
    Currently traj_pad_mask was removed. Word mask will be removed later.
    So nsd[i] = (
        (trajectory_features, k_key_tokens, decoder_in_char_seq, word_pad_mask), decoder_out_char_seq)

    ! It seems reasonable for the dataset to always return grid_name as a
    dict property. We just won't use it in collate function.
    """

    def __init__(self,
                 data_path: str,
                 gridname_to_grid: dict,
                 kb_tokenizer: KeyboardTokenizerv1,
                 word_tokenizer: CharLevelTokenizerv2,  # should contain max word len
                 include_time: bool = False,
                 include_velocities: bool = True,
                 include_accelerations: bool = True,
                 include_grid_name: bool = False,
                 has_target: bool = True,
                 has_one_grid_only: bool = True,
                 keyboard_selection_set: Optional[Set[str]] = None,
                 total: Optional[int] = None):
        """
        Arguments:
        ----------
        data_path: str
            Path to the NeuroSwipe dataset in JSON format.
            A custom version of the dataset is used: "grid" property
            is replaced with "grid_name". The grid itself is stored in
            a separate gridname_to_grid dictionary.
            Dataset is a list of JSON lines. Each line is a dictionary
            with the following properties:
            - word (str): word that was typed. May be absent if has_target is False.
            - curve (dict): dictionary that contains the following properties:
                - x (List[int]): x coordinates of the swipe trajectory.
                - y (List[int]): y coordinates of the swipe trajectory.
                - t (List[int]): time in milliseconds from the beginning of the swipe.
                - grid_name (str): name of the keyboard grid.

        gridname_to_grid: dict
            Dictionary that maps grid_name to grid.
            Grid is a dictionary that contains the following properties:
                - width (int): width of the keyboard in pixels.
                - height (int): height of the keyboard in pixels.
                - keys (List[dict]): list of keys. Each key is a dictionary
                    that contains the following properties:
                    - label (str): label of the key. May be absent if the key
                        is a special key (e.g. backspace).
                    - action (str): action of the key. May be absent if the key
                        is a character key (e.g. 'a', 'б', 'в').
                    - hitbox (dict): dictionary that contains the following properties:
                        - x (int): x coordinate of the top left corner of the key.
                        - y (int): y coordinate of the top left corner of the key.
                        - w (int): width of the key.
                        - h (int): height of the key.
            
        
        keyboard_selection_set: Optional[Set[str]]
            Set of keyboard key labels allowed. When looking
            for a key with the nearest to to trajectory point
            center coordinates we only consider keys with labels
            from this set.
            If None, all keys are allowed.
            Isn't used explicitly: only in is_allowed_label method.

        
        total: Optional[int]
            Number of dataset elements. Is used only for progress bar.

        """
        if include_accelerations and not include_velocities:
            raise ValueError("Accelerations are supposed \
                             to be an addition to velocities. Add velocities.")
        
        if has_one_grid_only and len(gridname_to_grid) != 1:
            raise ValueError(f"has_one_grid_only is True \
                             but len(gridname_to_grid) != 1")

        self.include_velocities = include_velocities
        self.include_accelerations = include_accelerations
        self.include_time = include_time
        self.has_target = has_target
        self.include_grid_name = include_grid_name
        self._keyboard_selection_set = keyboard_selection_set

        self.word_tokenizer = word_tokenizer

        self._grid_name_to_grid = gridname_to_grid

        self._nearest_kb_label_dict = (
            self._create_nearest_kb_label_dict(gridname_to_grid))

        self.data_list = []
        self._set_data(data_path, gridname_to_grid,
                       kb_tokenizer, self.data_list, total = total)


    def is_allowed_label(self, label: str) -> bool:
        if self._keyboard_selection_set is None:
            return True
        return label in self._keyboard_selection_set


    def get_nearest_kb_label(self, x, y, grid_name, gridname_to_grid):
        """
        Given coords on a keyboard (x, y) and its grid_name returns the nearest keyboard key

        By default it uses an array assosiated with grid_name
        that stores the nearest key label for every possible coord pair.

        If coords are outside of the keyboard boarders finds
        the nearest key by iterating over all keys.
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
        raise ValueError("Key has no label or action property")


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


    def _create_nearest_kb_label_dict(self, gridname_to_grid: dict
                                   ) -> Dict[str, np.array]:
        """
        Creates a dict that maps grid_name to a map (np.array)
        from coordinates [x, y] to nearest key label.
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
                   T: torch.tensor) -> List[float]:
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

        X = torch.tensor(X_list, dtype=torch.float32)
        Y = torch.tensor(Y_list, dtype=torch.float32)
        T = torch.tensor(T_list, dtype=torch.float32)

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

        if self.include_velocities:
            dx_dt = self._get_dx_dt(X, T)
            dy_dt = self._get_dx_dt(Y, T)
            xyt = torch.cat(
                [
                    xyt,
                    dx_dt.reshape(-1, 1),
                    dy_dt.reshape(-1, 1)
                ],
                axis = 1
            )

        if self.include_accelerations:
            d2x_dt2 = self._get_dx_dt(dx_dt, T)
            d2y_dt2 = self._get_dx_dt(dy_dt, T)
            xyt = torch.cat(
                [
                    xyt,
                    d2x_dt2.reshape(-1, 1),
                    d2y_dt2.reshape(-1, 1)
                ],
                axis = 1
            )
        
        
        grid = self._grid_name_to_grid[grid_name]
        xyt[:len(X_list), 0] = xyt[:len(X_list), 0] / grid['width'] 
        xyt[:len(Y_list), 1] = xyt[:len(X_list), 1] / grid['height']
        # Switch to this:
        # xyt[:, 0] = xyt[:, 0] / grid['width'] 
        # xyt[:, 1] = xyt[:, 1] / grid['height']

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
        
        if self.include_grid_name:
            return (xyt, kb_tokens, decoder_in_char_seq, word_mask), decoder_out_char_seq, grid_name
        
        return (xyt, kb_tokens, decoder_in_char_seq, word_mask), decoder_out_char_seq



class NeuroSwipeGridSubset(Dataset):
    def __init__(self, dataset: Dataset, grid_name: str):
        self.dataset = dataset
        self.grid_name = grid_name
        self.grid_name_idxs = self._get_grid_name_idxs()
        
            
    def _get_grid_name_idxs(self):
        grid_name_idxs: list[int] = []
        for i, (x, y, grid_name) in enumerate(self.dataset):
            if grid_name == self.grid_name:
                grid_name_idxs.append(i)
        return grid_name_idxs

    
    def __len__(self):
        return len(self.grid_name_idxs)
    
    def __getitem__(self, idx):
        return self.dataset[self.grid_name_idxs[idx]]
    


def collate_fn(batch: list):
    """
    batch - list of tuples:
    ((traj_feats, kb_tokens, dec_in_char_seq, word_pad_mask), dec_out_char_seq)
    """
    x, dec_out_char_seq = zip(*batch)
    (traj_feats, kb_tokens, dec_in_char_seq, word_pad_mask) = zip(*x)

    # traj_feats[i].shape = (curve_len, n_coord_feats)
    traj_feats = pad_sequence(traj_feats, batch_first=False)  # (curves_len, batch_size, n_coord_feats)
    # kb_tokens[i].shape = (curve_len,) 
    kb_tokens = pad_sequence(kb_tokens, batch_first=False)  # (curves_len, batch_size)
    
    dec_in_char_seq = torch.stack(dec_in_char_seq).transpose_(0, 1)  # (chars_seq_len - 1, batch_size)
    dec_out_char_seq = torch.stack(dec_out_char_seq).transpose_(0, 1)  # (chars_seq_len - 1, batch_size)
    word_pad_mask = torch.stack(word_pad_mask)
    

    max_curve_len = traj_feats.shape[0]

    traj_lens = torch.tensor([len(x) for x in traj_feats])
    

    # Берем матрицу c len(traj_lens) строками вида [0, 1, ... , max_curve_len - 1].
    # Каждый элемент i-ой строки сравниваем с длиной i-ой траектории.  Получится
    # матрица, где True только на позициях, больших, чем длина соответствующей траектории.
    # ! Возможно, можно проще
    traj_pad_mask = torch.arange(max_curve_len).expand(len(traj_lens), max_curve_len) >= traj_lens.unsqueeze(1)  # (batch_size, max_curve_len)    

    return (traj_feats, kb_tokens, dec_in_char_seq, traj_pad_mask, word_pad_mask), dec_out_char_seq
