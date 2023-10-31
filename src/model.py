import json
from typing import Optional, List, Tuple, Dict
import array

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class NeuroSwipeDatasetv1(Dataset):
    """
    Dataset class for NeuroSwipe dataset.
    The dataset file weights over 3 GB and contains over 6 million swipe gestures.
    """

    def __init__(self,
                 data_path: str,
                 grid: dict,
                 kb_tokenizer,
                 max_traj_len: int,
                 word_tokenizer,  # should contain max word len
                 include_time: bool = True,
                 include_velocities: bool = True,
                 include_accelerations: bool = True,
                 total: Optional[int] = None):
        """
        Args:
            data_path (string): Path to the NeuroSwipe dataset in JSON format.
                A custom version of the dataset is used:
                "grid" property is replaced with "grid_name" property.
        """
        if include_accelerations and not include_velocities:

            raise ValueError("Accelerations are supposed \
                             to be an addition to velocities. Add velocities.")

        self.max_traj_len = max_traj_len
        self.include_velocities = include_velocities
        self.include_accelerations = include_accelerations
        self.include_time = include_time

        self.word_tokenizer = word_tokenizer

        kb_keys = grid['keys']

        self.kb_width = grid['width']
        self.kb_height = grid['height']

        self.data_list = []
        self._set_data(data_path, kb_keys, kb_tokenizer, self.data_list, total = total)
    

    def _get_key_center(self, hitbox: Dict[str, int]) -> Tuple[int, int]:
        x = hitbox['x'] + hitbox['w'] / 2
        y = hitbox['y'] + hitbox['h'] / 2
        return x, y

    def _coord_to_kb_label(self, x: int, y:int, keys: List[dict]) -> str:
        nearest_kb_label = None
        min_dist = float("inf")
        for key in keys:
            key_x, key_y = self._get_key_center(key['hitbox'])
            dist = (x - key_x)**2 + (y - key_y)**2
            if dist < min_dist:
                min_dist = dist
                if 'label' in key:
                    nearest_kb_label = key['label']
                elif 'action' in key:
                    nearest_kb_label = key['action']  # tokenizer will covert it to <unk>
                else:
                    raise ValueError("Key has no label or action")

        return nearest_kb_label
            

    def _set_data(self,
                  data_path: str,
                  kb_keys: str,
                  kb_tokenizer,
                  data_list: list,
                  total: Optional[int] = None):
        with open(data_path, "r", encoding="utf-8") as json_file:
            for line in tqdm(json_file, total = total):
                data_list.append(self._get_data_from_json_line(line, kb_keys, kb_tokenizer))


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

    def _get_data_from_json_line(self, line, kb_keys, kb_tokenizer) -> Tuple[list, list, list, str]:
        """
        Parses a JSON line and returns a dictionary with data.
        """
        data = json.loads(line)
        word: str = data['word']

        X = array.array('h', data['curve']['x'])
        Y = array.array('h', data['curve']['y'])
        T = array.array('h', data['curve']['t'])        

        kb_labels = [self._coord_to_kb_label(x, y, kb_keys) for x,y in zip(X, Y)]
        kb_tokens = [kb_tokenizer.get_token(label) for label in kb_labels]
        kb_tokens += [kb_tokenizer.get_token('<pad>')] * (self.max_traj_len - len(kb_labels))
        kb_tokens = array.array('h', kb_tokens)

        return X, Y, T, word, kb_tokens

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        X_list, Y_list, T_list, word, kb_tokens = self.data_list[idx]

        X = torch.zeros(self.max_traj_len, dtype=torch.float32)
        Y = torch.zeros(self.max_traj_len, dtype=torch.float32)
        T = torch.zeros(self.max_traj_len, dtype=torch.float32)
        
        X[:len(X_list)] = torch.tensor(X_list, dtype=torch.float32) / self.kb_width
        Y[:len(Y_list)] = torch.tensor(Y_list, dtype=torch.float32) / self.kb_height
        T[:len(T_list)] = torch.tensor(T_list, dtype=torch.float32)

        xyt = torch.cat(
            [
                X.reshape(-1, 1),
                Y.reshape(-1, 1),
            ],
            axis = 1
        )

        if self.include_time:
            xyt = torch.cat(
                [
                    xyt,
                    T.reshape(-1, 1)
                ],
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

        char_seq, word_mask = self.word_tokenizer.tokenize(word)

        kb_tokens = torch.tensor(kb_tokens, dtype=torch.int64)
    
        return xyt, kb_tokens, traj_pad_mask, char_seq, word_mask