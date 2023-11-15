from typing import Callable
import os
import json
import pickle


import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from model import get_m1_model, get_m1_bigger_model, get_m1_smaller_model
from tokenizers import CharLevelTokenizerv2
from dataset import NeuroSwipeDatasetv2, NeuroSwipeGridSubset
from tokenizers import KeyboardTokenizerv1
from word_generation_v2 import predict_raw_mp
from word_generators import BeamGenerator


def get_grid(grid_name: str, grids_path: str) -> dict:
    with open(grids_path, "r", encoding="utf-8") as f:
        return json.load(f)[grid_name]


def weights_to_raw_predictions(grid_name: str,
                               model_getter: Callable,
                               weights_path: str,
                               word_char_tokenizer: CharLevelTokenizerv2,
                               dataset: Dataset,
                               generator_ctor,
                               n_workers: int = 4,
                               generator_kwargs = None
                               ):
     DEVICE = torch.device('cpu')  # Avoid multiprocessing with GPU
     if generator_kwargs is None:
          generator_kwargs = {}

     model = model_getter(DEVICE, weights_path)
     grid_name_to_greedy_generator = {grid_name: generator_ctor(model, word_char_tokenizer, DEVICE)}
     raw_predictions = predict_raw_mp(dataset,
                                      grid_name_to_greedy_generator,
                                      num_workers=n_workers,
                                      generator_kwargs=generator_kwargs)
     return raw_predictions


if __name__ == '__main__':

    DATA_ROOT = "../data/data_separated_grid"
    MODELS_ROOT = "../data/trained_models"
    DATASET_PATH = os.path.join(DATA_ROOT, 'test.jsonl')

    PREDS_ROOT = "../data/saved_beamsearch_results"

    MAX_WORD_LEN = 34  # len('информационно-телекоммуникационной')
    MAX_TRAJ_LEN = 299

    
    grid_name_to_grid__path = os.path.join(DATA_ROOT, "gridname_to_grid.json")
    grid_name_to_grid = {
        grid_name: get_grid(grid_name, grid_name_to_grid__path)
        for grid_name in ("default", "extra")
    }


    kb_tokenizer = KeyboardTokenizerv1()
    word_char_tokenizer = CharLevelTokenizerv2(
        os.path.join(DATA_ROOT, "voc.txt"))
    keyboard_selection_set = set(kb_tokenizer.i2t)

    print("Loading dataset...")
    dataset = NeuroSwipeDatasetv2(
        data_path = DATASET_PATH,
        gridname_to_grid = grid_name_to_grid,
        kb_tokenizer = kb_tokenizer,
        max_traj_len = MAX_TRAJ_LEN,
        word_tokenizer = word_char_tokenizer,
        include_time = False,
        include_velocities = True,
        include_accelerations = True,
        has_target=False,
        has_one_grid_only=False,
        include_grid_name=True,
        keyboard_selection_set=keyboard_selection_set,
        total = 10_000
    )

    grid_name_to_dataset = {
        'default': NeuroSwipeGridSubset(dataset, grid_name='default'),
        'extra': NeuroSwipeGridSubset(dataset, grid_name='extra'),
    }

    generator_kwargs = {
        'max_steps_n': MAX_WORD_LEN + 1,  # including '<eos>'
        'return_hypotheses_n': 7,
        'beamsize': 6,
        'normalization_factor': 0.5,
        'verbose': False
    }


    model_params = [
        ("default", get_m1_bigger_model, "m1_bigger/m1_bigger_v2__2023_11_12__14_51_49__0.13115__greed_acc_0.86034__default_l2_0_ls0_switch_2.pt"),
        ("default", get_m1_bigger_model, "m1_bigger/m1_bigger_v2__2023_11_12__12_30_29__0.13121__greed_acc_0.86098__default_l2_0_ls0_switch_2.pt"),
        ("default", get_m1_bigger_model, "m1_bigger/m1_bigger_v2__2023_11_11__22_18_35__0.13542_default_l2_0_ls0_switch_1.pt"),
        ("default", get_m1_model, "m1_v2/m1_v2__2023_11_09__10_36_02__0.14229_default_switch_0.pt"),
        ("default", get_m1_bigger_model, "m1_bigger/m1_bigger_v2__2023_11_12__00_39_33__0.13297_default_l2_0_ls0_switch_1.pt"),
        ("default", get_m1_bigger_model, "m1_bigger/m1_bigger_v2__2023_11_11__14_29_37__0.13679_default_l2_0_ls0_switch_0.pt"),

        ("extra", get_m1_model, "m1_v2/m1_v2__2023_11_09__17_47_40__0.14301_extra_l2_1e-05_switch_0.pt"),
        ("extra", get_m1_bigger_model, "m1_bigger/m1_bigger_v2__2023_11_12__02_27_14__0.13413_extra_l2_0_ls0_switch_1.pt"),
    ]


    for grid_name, model_getter, weights_f_name in model_params:
        assert os.path.exists(os.path.join(MODELS_ROOT, weights_f_name)), \
            f"Path {weights_f_name} does not exist."


    for grid_name, model_getter, weights_f_name in model_params:

        bs_preds_path = os.path.join(PREDS_ROOT,
                                     f"{weights_f_name.replace('/', '__')}.pkl")
        
        if os.path.exists(bs_preds_path):
            print(f"Path {bs_preds_path} exists. Skipping.")
            continue

        bs_predictions = weights_to_raw_predictions(
            grid_name = grid_name,
            model_getter=model_getter,
            weights_path = os.path.join(MODELS_ROOT, weights_f_name),
            word_char_tokenizer=word_char_tokenizer,
            dataset=grid_name_to_dataset[grid_name],
            generator_ctor=BeamGenerator,
            n_workers=4,
            generator_kwargs=generator_kwargs
        )

        with open(bs_preds_path, 'wb') as f:
            pickle.dump(bs_predictions, f, protocol=pickle.HIGHEST_PROTOCOL)
