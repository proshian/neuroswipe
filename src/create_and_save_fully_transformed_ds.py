# command:
# python ./src/create_and_save_fully_transformed_ds.py --jsonl_path ./data/data_separated_grid/train__default_only_no_errors__2023_10_31__03_26_16.jsonl --output_path ./train_default_grid_no_errors__2023_10_31_ft__uint8 --vocab_path ./data/data_separated_grid/voc.txt --gridname_to_grid_path ./data/data_separated_grid/gridname_to_grid.json --n_workers 0

import os; import sys; current_dir = os.path.dirname(os.path.abspath(__file__)); parent_dir = os.path.dirname(current_dir); sys.path.append(os.path.join(parent_dir, 'src'))


import pickle
from abc import ABC, abstractmethod
from math import ceil


import argparse
import torch
from tqdm import tqdm 

from dataset import CurveDatasetWithMultiProcInit
from nearest_key_lookup import NearestKeyLookup
from tokenizers import KeyboardTokenizerv1, CharLevelTokenizerv2
from tokenizers import ALL_CYRILLIC_LETTERS_ALPHABET_ORD
from predict import get_grid_name_to_grid
# from transforms import InitTransform, GetItemTransform
from transforms import FullTransform



class ListStorage(ABC):
    @abstractmethod
    def read(self, path):
        raise NotImplementedError()
    
    @abstractmethod
    def write(self, path, data):
        raise NotImplementedError()
    

class BinsListStorage(ListStorage):
    """
    Stores list in a folder where each file is 
    a sub-list of the original list of size bin_size
    """
    def __init__(self, bin_size: int):
        self.bin_size = bin_size

    def _check_all_files_exist(self, path, n_bins):
        for i in range(n_bins):
            assert os.path.isfile(os.path.join(path, f'{i}.pkl'))

    def read(self, path):
        assert os.path.isdir(path)
        with open(os.path.join(path, 'meta_info.pkl'), 'rb') as f:
            meta_info = pickle.load(f)
        n_bins = ceil(meta_info['data_len'] / self.bin_size)
        self._check_all_files_exist(path, n_bins)
        data = [None] * meta_info['data_len']
        for i in tqdm(range(n_bins)):
            with open(os.path.join(path, f'{i}.pkl'), 'rb') as f:
                data[i * self.bin_size: (i + 1) * self.bin_size] = pickle.load(f)
        return data

    
    def write(self, path, data: list):
        os.makedirs(path, exist_ok=False)
        n_bins = ceil(len(data) / self.bin_size)
        for i in tqdm(range(n_bins)):
            with open(os.path.join(path, f'{i}.pkl'), 'wb') as f:
                pickle.dump(data[i * self.bin_size: (i + 1) * self.bin_size], f)
        meta_info = {'data_len': len(data), 'bin_size': self.bin_size}
        with open(os.path.join(path, 'meta_info.pkl'), 'wb') as f:
            pickle.dump(meta_info, f)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create and save dataset that stores ' \
            'fully transformed (model_input, target)')
    parser.add_argument('--jsonl_path', type=str, help='Path to the jsonl dataset file')
    parser.add_argument('--output_path', type=str, help='Path to save the fully transformed dataset')
    parser.add_argument('--vocab_path', type=str, help='Path to the vocabulary file')
    parser.add_argument('--gridname_to_grid_path', type=str, help='Path to the gridname_to_grid file')
    parser.add_argument('--n_workers', type=int, help='Number of workers to use')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    gridname_to_grid = get_grid_name_to_grid(args.gridname_to_grid_path)

    print("Creating nearest key lookup for each grid...")
    gridname_to_nkl = {
        gname: NearestKeyLookup(grid, ALL_CYRILLIC_LETTERS_ALPHABET_ORD)
        for gname, grid in gridname_to_grid.items()
    }

    gname_to_wh = {
        gname: (grid['width'], grid['height']) 
        for gname, grid in gridname_to_grid.items()
    }

    kb_tokenizer = KeyboardTokenizerv1()
    word_tokenizer = CharLevelTokenizerv2(args.vocab_path)


    full_transform = FullTransform(
        grid_name_to_nk_lookup=gridname_to_nkl,
        grid_name_to_wh=gname_to_wh,
        kb_tokenizer=kb_tokenizer,
        word_tokenizer=word_tokenizer,
        include_time=False,
        include_velocities=True,
        include_accelerations=True,
        kb_tokens_dtype=torch.uint8,
        word_tokens_dtype=torch.uint8
    )

    print("Calling CurveDatasetWithMultiProcInit.__init__ with full_transform...")
    ds = CurveDatasetWithMultiProcInit(
        data_path=args.jsonl_path,
        store_gnames=False,
        init_transform=full_transform,
        get_item_transform=None,
        n_workers=args.n_workers,
        total=6_000_000
    )

    print("saving")

    bls = BinsListStorage(bin_size=10_000)
    bls.write(args.output_path, ds.data_list)

    # torch.save(ds.data_list, args.output_path)
    # with open(args.output_path, 'wb') as f:
    #     pickle.dump(ds.data_list, f)
    
    # The script craches without error message sometimes and there's no way
    # to know that it succeeded without this print
    print("saved")