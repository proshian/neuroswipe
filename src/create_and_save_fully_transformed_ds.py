# command:
# python ./src/create_and_save_fully_transformed_ds.py --jsonl_path ./data/data_separated_grid/train__default_only_no_errors__2023_10_31__03_26_16.jsonl --output_path ./train_default_grid_no_errors__2023_10_31_ft__int32.pkl --vocab_path ./data/data_separated_grid/voc.txt --gridname_to_grid_path ./data/data_separated_grid/gridname_to_grid.json --n_workers 0

import os; import sys; current_dir = os.path.dirname(os.path.abspath(__file__)); parent_dir = os.path.dirname(current_dir); sys.path.append(os.path.join(parent_dir, 'src'))


import argparse
import pickle

from dataset import CurveDatasetWithMultiProcInit
from nearest_key_lookup import NearestKeyLookup
from tokenizers import KeyboardTokenizerv1, CharLevelTokenizerv2
from tokenizers import ALL_CYRILLIC_LETTERS_ALPHABET_ORD
from predict import get_grid_name_to_grid
# from transforms import InitTransform, GetItemTransform
from transforms import TransformerInputOutputGetter


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


    full_transform = TransformerInputOutputGetter(
        grid_name_to_nk_lookup=gridname_to_nkl,
        grid_name_to_wh=gname_to_wh,
        kb_tokenizer=kb_tokenizer,
        word_tokenizer=word_tokenizer,
        include_time=False,
        include_velocities=True,
        include_accelerations=True
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

    # ! Probably better to use torch.save 
    with open(args.output_path, 'wb') as f:
        pickle.dump(ds.data_list, f)