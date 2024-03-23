# commands:
# python ./src/save_fully_transformed_ds.py --jsonl_path ./data/data_separated_grid/train__default_only_no_errors__2023_10_31__03_26_16.jsonl --output_path ./train_default_grid_no_errors__2023_10_31_ft__uint8 --vocab_path ./data/data_separated_grid/voc.txt --gridname_to_grid_path ./data/data_separated_grid/gridname_to_grid.json --n_workers 0

# python ./src/save_fully_transformed_ds.py --jsonl_path ./data/data_separated_grid/train__extra_only_no_errors__2023_11_01__19_49_14.jsonl --output_path ./train_extra_no_errors_uint8_datalist.pt --vocab_path ./data/data_separated_grid/voc.txt --gridname_to_grid_path ./data/data_separated_grid/gridname_to_grid.json --n_workers 0


# import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "src"))


import pickle
from abc import ABC, abstractmethod
from math import ceil
import os
import sys

import argparse
import torch
from tqdm import tqdm 

from dataset import CurveDatasetWithMultiProcInit
from nearest_key_lookup import NearestKeyLookup
from tokenizers import KeyboardTokenizerv1, CharLevelTokenizerv2
from tokenizers import ALL_CYRILLIC_LETTERS_ALPHABET_ORD
from predict import get_grid_name_to_grid
from transforms import FullTransform, TrajFeatsKbTokensTgtWord_InitTransform
from dataset import _get_data_from_json_line


class ListStorage(ABC):
    """
    Saves and loads list from disk
    """
    @abstractmethod
    def load(self, path):
        raise NotImplementedError()
    
    @abstractmethod
    def save(self, path, data):
        raise NotImplementedError()
    

class BinsListStorage(ListStorage):
    """
    Stores list in a folder where each file is 
    a sub-list of the original list of size bin_size
    """
    def _check_all_files_exist(self, path, n_bins):
        for i in range(n_bins):
            assert os.path.isfile(os.path.join(path, f'{i}.pkl'))

    def load(self, path) -> list:
        assert os.path.isdir(path)

        with open(os.path.join(path, 'meta_info.pkl'), 'rb') as f:
            meta_info = pickle.load(f)
        bin_size = meta_info['bin_size']
        n_bins = ceil(meta_info['data_len'] / bin_size)

        self._check_all_files_exist(path, n_bins)

        data = [None] * meta_info['data_len']
        for i in tqdm(range(n_bins)):
            with open(os.path.join(path, f'{i}.pkl'), 'rb') as f:
                data[i * bin_size: (i + 1) * bin_size] = pickle.load(f)
        
        assert len(data) == meta_info['data_len']

        return data

    def save(self, path, bin_size, data: list) -> None:
        os.makedirs(path, exist_ok=False)
        n_bins = ceil(len(data) / bin_size)
        meta_info = {'data_len': len(data), 'bin_size': bin_size}
        with open(os.path.join(path, 'meta_info.pkl'), 'wb') as f:
            pickle.dump(meta_info, f)
        for i in tqdm(range(n_bins)):
            with open(os.path.join(path, f'{i}.pkl'), 'wb') as f:
                pickle.dump(data[i * bin_size: (i + 1) * bin_size], f)


def move_list_to_disk_with_delete(bin_size, path, data: list) -> None:
    """
    Similar to BinsListStorage.write, but deletes list.
    This function is used to not overflow memory. It calls 
    data.pop() to accumulate elements for next bin 
    and free memory at the same time.
    """
    os.makedirs(path, exist_ok=False)
    initial_data_len = len(data)
    n_bins = ceil(initial_data_len / bin_size)
    meta_info = {'data_len': len(data), 'bin_size': bin_size}
    with open(os.path.join(path, 'meta_info.pkl'), 'wb') as f:
        pickle.dump(meta_info, f)
    for i in tqdm(range(n_bins - 1, -1, -1)):
        bin = data[i * bin_size: (i + 1) * bin_size]
        for j in range(len(bin) -1, -1, -1):
            last_data_el = data.pop()
            assert last_data_el == bin[j]
        with open(os.path.join(path, f'{i}.pkl'), 'wb') as f:
            pickle.dump(bin, f)
    assert len(data) == 0


def jsonl_to_bins_on_disk(jsonl_path, bin_size, 
                          transform, out_path, total = None) -> None:
    os.makedirs(out_path, exist_ok=False)
    bin = []
    n_bins = 0
    n_elements = 0
    with open(jsonl_path, "r", encoding="utf-8") as json_file:
        for line in tqdm(json_file, total = total):
            data_el = transform(_get_data_from_json_line(line))
            bin.append(data_el)
            if len(bin) == bin_size:
                with open(os.path.join(out_path, f'{n_bins}.pkl'), 'wb') as f:
                    pickle.dump(bin, f)
                n_bins += 1
                n_elements += bin_size
                bin = []
                
    if bin:
        with open(os.path.join(out_path, f'{n_bins}.pkl'), 'wb') as f:
            pickle.dump(bin, f)
        n_bins += 1
        n_elements += len(bin)
    
    meta_info = {'data_len': n_elements, 'bin_size': bin_size}
    with open(os.path.join(out_path, 'meta_info.pkl'), 'wb') as f:
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


    transform = FullTransform(
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


    # transform = TrajFeatsKbTokensTgtWord_InitTransform(
    #     grid_name_to_nk_lookup=gridname_to_nkl,
    #     grid_name_to_wh=gname_to_wh,
    #     kb_tokenizer=kb_tokenizer,
    #     include_time=False,
    #     include_velocities=True,
    #     include_accelerations=True,
    # )

    print("Calling CurveDatasetWithMultiProcInit.__init__ with full_transform...")


    # jsonl_to_bins_on_disk(args.jsonl_path, 10_000, transform, args.output_path, total=5_237_584)



    ds = CurveDatasetWithMultiProcInit(
        data_path=args.jsonl_path,
        store_gnames=False,
        init_transform=transform,
        get_item_transform=None,
        n_workers=args.n_workers,
        total=5_237_584
    )

    # This was a bad idea because it's size of list 
    # and the list seems to store links, not values...
    # print(f"{sys.getsizeof(ds.data_list) = }")

    print("saving")



    # move_list_to_disk_with_delete(10_000, args.output_path, ds.data_list)




    torch.save(ds.data_list, args.output_path)
    # with open(args.output_path, 'wb') as f:
    #     pickle.dump(ds.data_list, f)
    
    # The script craches without error message sometimes and there's no way
    # to know that it succeeded without this print
    print("saved")
    