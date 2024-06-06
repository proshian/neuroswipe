import json
from typing import Optional, List, Tuple, Callable
import array
from multiprocessing import Pool


import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


RawDatasetEl = Tuple[array.array, array.array, 
                     array.array, str, Optional[str]]


def _get_data_from_json_line(line) -> RawDatasetEl:
    data = json.loads(line)

    X = array.array('h', data['curve']['x'])
    Y = array.array('h', data['curve']['y'])
    T = array.array('h', data['curve']['t'])

    grid_name = data['curve']['grid_name']   

    tgt_word = data['word'] if 'word' in data else None

    return X, Y, T, grid_name, tgt_word


class CurveDataset(Dataset):
    """
    Dataset class for NeuroSwipe jsonl dataset
    
    if `init_transform` and `get_item_transform` are None, 
    curve_dataset_obj[i] is a tuple (X, Y, T, grid_name, tgt_word)
    If there is no 'word' property in .json file, `tgt_word` is None.

    Transforms are separated into two parts: 
    * `init_transform` - takes raw data and returns semi-extracted features 
    * `get_item_transform` takes semi-extracted features and returns 
        (model_input, target).
    
    If `init_transform` is a full transform, the dataset may take too much memory.
    If `get_item_transform` is a full transform, iterating over 
        the dataset may be slow. 
    """

    def __init__(self,
                 data_path: str,
                 store_gnames: bool,
                 init_transform: Optional[Callable] = None,
                 get_item_transform: Optional[Callable] = None,
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
            - word (str): word that was typed. 
                Is abscent in test and val datasets.
            - curve (dict): dictionary that contains the following properties:
                - x (List[int]): x coordinates of the swipe trajectory.
                - y (List[int]): y coordinates of the swipe trajectory.
                - t (List[int]): time (in ms) from the beginning of the swipe.
                - grid_name (str): name of the keyboard grid.
        store_gnames: bool
            If True, stores grid names in self.grid_name_list.
        init_transform: Optional[Callable]
            A function that takes raw data (X, Y, T, grid_name, tgt_word)
            and returns semi-extracted features.
        get_item_transform: Optional[Callable]
            A function that takes semi-extracted features and returns 
            (model_input, target).
        total: Optional[int]
            Number of dataset elements. Is used only for progress bar.
        """
        self.transform = get_item_transform
        self.data_list = self._get_data(
            data_path, init_transform, store_gnames, total)
        
    def _get_data(self,
                  data_path: str,
                  transform: Optional[Callable],
                  set_gnames: bool,
                  total: Optional[int] = None) -> List[RawDatasetEl]:
        data_list = []
        if set_gnames:
            self.grid_name_list = []
        with open(data_path, "r", encoding="utf-8") as json_file:
            for line in tqdm(json_file, total = total):
                data_el = self._get_data_from_json_line(line)
                if set_gnames:
                    self.grid_name_list.append(data_el[3])
                if transform is not None:
                    data_el = transform(data_el)
                data_list.append(data_el)
        return data_list

    def _get_data_from_json_line(self,
                                 line
                                 ) -> RawDatasetEl:
        return _get_data_from_json_line(line)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample = self.data_list[idx]  # X, Y, T, grid_name, tgt_word
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    @classmethod
    def from_data_list(cls, 
                       data_list: list, 
                       grid_name_list: Optional[List[str]] = None,
                       get_item_transform: Optional[Callable] = None,
                       ):
        if grid_name_list:
            assert len(grid_name_list) == len(data_list)
        
        obj = cls.__new__(cls)

        obj.data_list = data_list
        obj.transform = get_item_transform

        if grid_name_list:
            obj.grid_name_list = grid_name_list

        return obj


class CurveDatasetWithMultiProcInit(CurveDataset):
    def __init__(self,
                 data_path: str,
                 store_gnames: bool,
                 init_transform: Optional[Callable] = None,
                 get_item_transform: Optional[Callable] = None,
                 n_workers: int = 0,
                 total: Optional[int] = None):
        """
        Arguments:
        ----------
        **All arguments from CurveDatase are present and are same**.
        n_workers: int
            If `n_workers` > 0, dataset creation will be parallelized.
        """
        self.n_workers = n_workers
        self.transform = get_item_transform

        get_data_fn = self._get_data_mp if n_workers > 0 else self._get_data
        self.data_list = get_data_fn(data_path, init_transform, 
                                     store_gnames, total)
        
    def _get_data_mp(self,
                    data_path: str,
                    transform: Optional[Callable],
                    set_gnames: bool,
                    total: Optional[int] = None) -> List[RawDatasetEl]:
        data_list = []
        if set_gnames:
            self.grid_name_list = []
        with open(data_path, "r", encoding="utf-8") as json_file:
            with Pool(self.n_workers) as executor:
                # Seems like choosing proper chunk size is crucial for efficiency.
                # Seems like splitting the file into portions leads to overhead.
                # Note that processign speeds up a lot after around 10 minutes. 
                n_chunks_per_workser = 8
                chunksize = int(total / n_chunks_per_workser / self.n_workers)
                # cuncurrent.futures.PoolExecutor.map and Pool.map do not 
                # satisfy the task since they collect iterable immediately.
                for data_el in tqdm(executor.imap(self._get_data_from_json_line, json_file, chunksize = chunksize), total = total):
                    if set_gnames:
                        self.grid_name_list.append(data_el[3])
                    if transform is not None:
                        data_el = transform(data_el)
                    data_list.append(data_el)

        return data_list



class CurveDatasetSubset:
    def __init__(self, dataset: CurveDataset, grid_name: str):
        assert hasattr(dataset, 'grid_name_list'), \
            "Dataset doesn't have grid_name_list property. " \
            "To fix this create the dataset with store_gnames=True"
        # ! Maybe check dataset.grid_name_list is Iterable
        assert dataset.grid_name_list is not None
        assert len(dataset) == len(dataset.grid_name_list)
        
        self.dataset = dataset
        self.grid_name = grid_name
        self.grid_idxs = self._get_grid_idxs()
    
    def _get_grid_idxs(self):
        return [i for i, gname in enumerate(self.dataset.grid_name_list)
                if gname == self.grid_name]
    
    def __len__(self):
        return len(self.grid_idxs)
    
    def __getitem__(self, idx):
        return self.dataset[self.grid_idxs[idx]]
    

class CollateFn:
    def __init__(self, word_pad_idx: int, batch_first: bool):
        self.word_pad_idx = word_pad_idx
        self.batch_first = batch_first
    
    def __call__(self, batch: list):
        """
        Arguments:
        ----------
        batch: list of tuples:
            ((traj_feats, kb_tokens, dec_in_char_seq), dec_out_char_seq)
        """
        x, dec_out_no_pad = zip(*batch)
        (traj_feats_no_pad, kb_tokens_no_pad, dec_in_no_pad) = zip(*x)

        # traj_feats[i].shape = (curve_len, batch_size, n_coord_feats)
        traj_feats = pad_sequence(traj_feats_no_pad, batch_first=self.batch_first)
        # kb_tokens[i].shape = (curve_len, batch_size)
        kb_tokens = pad_sequence(kb_tokens_no_pad, batch_first=self.batch_first)

        # (chars_seq_len - 1, batch_size)
        dec_in = pad_sequence(dec_in_no_pad, batch_first=self.batch_first, 
                                       padding_value=self.word_pad_idx)
        dec_out = pad_sequence(dec_out_no_pad, batch_first=self.batch_first,
                                        padding_value=self.word_pad_idx)
        
        
        word_pad_mask = dec_in == self.word_pad_idx
        if not self.batch_first:
            word_pad_mask = word_pad_mask.T  # word_pad_mask is always batch first

        max_curve_len = traj_feats.shape[1] if self.batch_first else traj_feats.shape[0]
        traj_lens = torch.tensor([len(x) for x in traj_feats_no_pad])

        # Берем матрицу c len(traj_lens) строками вида 
        # [0, 1, ... , max_curve_len - 1].  Каждый элемент i-ой строки 
        # сравниваем с длиной i-ой траектории.  Получится матрица, где True 
        # только на позициях, больших, чем длина соответствующей траектории.
        # (batch_size, max_curve_len)    
        traj_pad_mask = torch.arange(max_curve_len).expand(
            len(traj_lens), max_curve_len) >= traj_lens.unsqueeze(1)
        
        transformer_in = (traj_feats, kb_tokens, dec_in, 
                          traj_pad_mask, word_pad_mask)

        return transformer_in, dec_out



class CollateFnV2:
    def __init__(self, batch_first: bool, word_pad_idx: int, 
                 swipe_pad_idx: int = 0) -> None:
        self.word_pad_idx = word_pad_idx
        self.batch_first = batch_first
        self.swipe_pad_idx = swipe_pad_idx

    def _assert_encoder_in_type_and_shape(self, encoder_in_example):
        assert len(encoder_in_example) == 2
        for el in encoder_in_example:
            assert isinstance(el, torch.Tensor), \
                f"Expected torch.Tensor, got {type(el)}"
        
    def _is_encoder_input_tuple(self, batch):
        encoder_in_example = batch[0][0][0]
        if isinstance(encoder_in_example, tuple):
            self._assert_encoder_in_type_and_shape(encoder_in_example)
            return True
        elif isinstance(encoder_in_example, torch.Tensor):
            return False
        else:
            raise ValueError(f"Unknown type of encoder input {type(batch[0][0])}")
    
    def __call__(self, batch: list):
        """
        Given a List where each row is 
        ((encoder_in_sample, decoder_in_sample), decoder_out_sample) 
        returns a tuple of two elements:
        1. (encoder_in, decoder_in, swipe_pad_mask, word_pad_mask)
        2. decoder_out

        Arguments:
        ----------
        batch: list of tuples:
            ((encoder_in, dec_in_char_seq), dec_out_char_seq),
            where encoder_in may be a tuple of torch tensors
            (ex. ```(traj_feats, nearest_kb_tokens)```)
            or a single tensor (ex. ```nearest_kb_tokens```)


        Returns:
        --------
        transformer_in: tuple of torch tensors:
            1. (enc_in, dec_in, swipe_pad_mask, word_pad_mask),
                where enc_in can be either a single tensor or a tuple
                of two tensors (depends on type of input)
                Each element is a torch tensor of shape:
                - enc_in: either (curve_len, batch_size, n_feats) or
                    ((curve_len, batch_size, n_feats1), (curve_len, batch_size, n_feats2))
                - dec_in: (chars_seq_len - 1, batch_size)
                - swipe_pad_mask: (batch_size, curve_len)
                - word_pad_mask: (batch_size, chars_seq_len - 1, )
        """
        is_encoder_input_tuple = self._is_encoder_input_tuple(batch)
        dec_in_no_pad = []
        dec_out_no_pad = []

        encoder_in_no_pad = ([], []) if is_encoder_input_tuple else []

        for row in batch:
            x_smpl, decoder_out_smpl = row
            encoder_in_smpl, decoder_in_smpl = x_smpl
            if is_encoder_input_tuple:
                for i in range(2):
                    encoder_in_no_pad[i].append(encoder_in_smpl[i])
            else:
                encoder_in_no_pad.append(encoder_in_smpl)

            dec_in_no_pad.append(decoder_in_smpl)
            dec_out_no_pad.append(decoder_out_smpl)

        if is_encoder_input_tuple:
            encoder_in = tuple(pad_sequence(encoder_in_no_pad_i, batch_first=self.batch_first, 
                                       padding_value=self.swipe_pad_idx)
                          for encoder_in_no_pad_i in encoder_in_no_pad)
        else:
            encoder_in = pad_sequence(encoder_in_no_pad, batch_first=self.batch_first,
                                      padding_value=self.swipe_pad_idx)

        dec_out = pad_sequence(dec_out_no_pad, batch_first=self.batch_first,
                                        padding_value=self.word_pad_idx)
        
        dec_in = pad_sequence(dec_in_no_pad, batch_first=self.batch_first,
                                        padding_value=self.word_pad_idx)
        
        word_pad_mask = dec_in == self.word_pad_idx
        if not self.batch_first:
            word_pad_mask = word_pad_mask.T  # word_pad_mask is always batch first

        encoder_in_el = encoder_in[0] if is_encoder_input_tuple else encoder_in
        max_curve_len = encoder_in_el.shape[1] if self.batch_first else encoder_in_el.shape[0]
        encoder_in_no_pad_el = encoder_in_no_pad[0] if is_encoder_input_tuple else encoder_in_no_pad
        encoder_lens = torch.tensor([len(x) for x in encoder_in_no_pad_el])

        # Берем матрицу c len(encoder_lens) строками вида
        # [0, 1, ... , max_curve_len - 1].  Каждый элемент i-ой строки
        # сравниваем с длиной i-ой траектории.  Получится матрица, где True
        # только на позициях, больших, чем длина соответствующей траектории.
        # (batch_size, max_curve_len)
        encoder_pad_mask = torch.arange(max_curve_len).expand(
            len(encoder_lens), max_curve_len) >= encoder_lens.unsqueeze(1)
        
        transformer_in = (encoder_in, dec_in, encoder_pad_mask, word_pad_mask)
        
        return transformer_in, dec_out
