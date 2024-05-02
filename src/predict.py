# Сейчас предсказания отдельных моделей сохраняются как список списков
# кортежей (-log(prob), pred_word).  Пусть модель заточена под раскладку
# клавиатуры grid_name.  Пусть dataset[grid_name] – это датасет,
# сохраняющий порядок исходного датасета, но исключающий все экземпляры
# с другой раскладкой.  Пусть предсказание хранится в переменной preds.
# Тогда preds[i] - это список предсказаний для кривой dataset[grid_name][i].
# Данный список представлен кортежами (-log(prob), pred_word).


# На входе скрипта предсказания:
# * модели, от которых мы хотим получить предсказания. Модели имеют:
#   * название раскладки
#   * название архитктуры
#   * путь к весам
# * Алгоритм декодирования слова и его аргументы
# * датасет в виде JSON, для которого хотим получить предсказания (точнее,
#       пердсказания хотим получить для поднабора этого датасета,
#       с клавитурами конкретной раскладки)
#
#
# Гипотетически могут быть модели, умеющие работать сразу с
# множеством раскладок.  Предсказание для таких моделей делается
# точно также, отдельно для каждой раскладки.


# Результат модуля предсказаний будет подан на вход скрипту аггрегации,
# а также обучения аггрегации.
# Вход аггрегации в рамках одной раскладки должен быть следующим:
# * список с элементами (pred_id, pred)
# * состояние аггрегатора
# * название раскладки
# В качестве pred_id может выступать
# f"{weights_path}__{generator_type}__{generator_kwargs}__{grid_name}".
# Лучше это будет буквально id, сохраненный где-то
# в отдельном файле / базе данных.



from typing import Iterable, List, Tuple, Dict, Optional
import os
import json
import pickle
import argparse
from dataclasses import dataclass, asdict


import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
# import pandas as pd

from model import MODEL_GETTERS_DICT
from ns_tokenizers import CharLevelTokenizerv2, KeyboardTokenizerv1
from dataset import CurveDataset, CurveDatasetSubset
from ns_tokenizers import ALL_CYRILLIC_LETTERS_ALPHABET_ORD
from word_generators import GENERATOR_CTORS_DICT
from transforms import KbTokens_InitTransform, KbTokens_GetItemTransform
from nearest_key_lookup import NearestKeyLookup



@dataclass
class Prediction:
    prediction: List[List[Tuple[float, str]]]
    model_name: str
    model_weights: str
    generator_name: str
    generator_kwargs: dict
    grid_name: str
    dataset_split: str


class Predictor:
    """
    Creates a prediction for a whole dataset.

    The resulting prediction contains metadata with information about
    model and decoding algorithm useful later for predictions aggregation.
    """

    def __init__(self,
                 model_architecture_name: str,
                 model_weights_path: str,
                 word_generator_type: str,
                 generator_kwargs: Optional[dict]=None,
                 ) -> None:
        DEVICE = torch.device('cpu')

        self.word_generator_type = word_generator_type
        word_generator_ctor = GENERATOR_CTORS_DICT[word_generator_type]
        self.model_architecture_name = model_architecture_name
        self.model_weights_path = model_weights_path
        model_getter = MODEL_GETTERS_DICT[model_architecture_name]
        model = model_getter(DEVICE, model_weights_path)
        self.word_char_tokenizer = CharLevelTokenizerv2(config['voc_path'])
        self.word_generator = word_generator_ctor(
            model, self.word_char_tokenizer, DEVICE)
        self.generator_kwargs = generator_kwargs

    def _predict_example(self,
                         data: Tuple[int, Tuple[Tensor, Tensor]]
                         ) -> Tuple[int, List[Tuple[float, str]]]:
        """
        Predicts a single example.

        Arguments:
        ----------
        data: Tuple[i, gen_in]
            i: int
                Index of the example in the dataset.
            gen_in: Tuple[Tensor, Tensor]
                Tuple of (traj_feats, kb_tokens)

        Returns:
        --------
        i: int
            Index of the example in the dataset.
        pred: List[Tuple[log_probability, char_sequence]]
            log_probability: float
            char_sequence: str
        """
        i, gen_in = data
        pred = self.word_generator(*gen_in, **self.generator_kwargs)
        return i, pred
    
    def _predict_raw_mp(self, dataset: CurveDataset,
                        num_workers: int=3) -> List[List[Tuple[float, str]]]:
        """
        Creates predictions given a word generator
        
        Arguments:
        ----------
        dataset: CurveDataset
            The dataset is supposed to be a subset of the original dataset
            containing only examples with the same grid_name as the predictor.
        num_workers: int
            Number of processes.

        Returns:
        --------
        preds: List[one_swipe_preds]
            one_swipe_preds: List[Tuple[log_probability, char_sequence]]
                log_probability: float
                char_sequence: str
        """
        preds = [None] * len(dataset)

        data = [(i, (traj_feats, kb_tokens))
                for i, ((traj_feats, kb_tokens, _), _) in enumerate(dataset)]     
        
        with ProcessPoolExecutor(num_workers) as executor:
            for i, pred in tqdm(executor.map(self._predict_example, data), total=len(dataset)):
                preds[i] = pred

        return preds

    def predict(self, dataset: CurveDataset, 
                grid_name: str, dataset_split: str,
                num_workers: int=3) -> Prediction:
        """
        Creates predictions given a word generator
        
        Arguments:
        ----------
        dataset: CurveDataset
            Output[i] is a list of predictions for dataset[i] curve.
        grid_name: str
        dataset_split: str
        num_workers: int
            Number of processes.
        """
        preds = self._predict_raw_mp(dataset, num_workers)

        preds_with_meta = Prediction(
            preds, self.model_architecture_name,
            self.model_weights_path, self.word_generator_type,
            self.generator_kwargs, grid_name, dataset_split)
        
        return preds_with_meta


# def create_new_df() -> pd.DataFrame:
#     df = pd.DataFrame(columns = [
#             'predictor_id', 'model_name', 'model_weights', 'generator_name', 'grid_name', 'dataset_split'])

# def load_df(preds_csv_path: str) -> pd.DataFrame:
#     if not os.path.exists(preds_csv_path):
#         print(f"Warning: {preds_csv_path} does not exist. Creating a new df...")
#         create_new_df()
    

def save_predictions(preds_wtih_meta: Prediction,
                     out_path: str,
                     preds_csv_path: str) -> None:
    with open(out_path, 'wb') as f:
        pickle.dump(
            preds_wtih_meta.prediction, f, protocol=pickle.HIGHEST_PROTOCOL)

#     # df = load_df(preds_csv_path)
#     # update_database(df, preds_wtih_meta)
#     # df.to_csv(preds_csv_path, index=False)


def get_grid(grid_name: str, grids_path: str) -> dict:
    with open(grids_path, "r", encoding="utf-8") as f:
        return json.load(f)[grid_name]


def get_grid_name_to_grid(grid_name_to_grid__path: str, 
                          allowed_gnames = ("default", "extra")) -> dict:
    # In case there will be more grids in "grid_name_to_grid.json"
    grid_name_to_grid = {
        grid_name: get_grid(grid_name, grid_name_to_grid__path)
        for grid_name in allowed_gnames
    }
    return grid_name_to_grid


def get_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        full_config = json.load(f)
    prediction_config = dict(full_config['prediction_config'])

    data_split = prediction_config['data_split']
    data_path = prediction_config['data_split__to__path'][data_split]
    prediction_config['data_path'] = data_path

    return prediction_config


def get_gridname_to_dataset(config) -> Dict[str, Dataset]:
    word_char_tokenizer = CharLevelTokenizerv2(config['voc_path'])

    kb_tokenizer = KeyboardTokenizerv1()
                                 
    grid_name_to_grid = get_grid_name_to_grid(
        config['grid_name_to_grid__path'])
    

    gname_to_wh = {
        gname: (grid['width'], grid['height']) 
        for gname, grid in grid_name_to_grid.items()
    }
    
    print("Creating NearestKeyLookups...")
    gridname_to_nkl = {
        gname: NearestKeyLookup(grid, ALL_CYRILLIC_LETTERS_ALPHABET_ORD)
        for gname, grid in grid_name_to_grid.items()
    }
    
    init_transform = KbTokens_InitTransform(
        grid_name_to_nk_lookup=gridname_to_nkl,
        kb_tokenizer=kb_tokenizer,
    )

    get_item_transform = KbTokens_GetItemTransform(
        grid_name_to_wh=gname_to_wh,
        word_tokenizer=word_char_tokenizer,
        include_time=False,
        include_velocities=True,
        include_accelerations=True,
    )

    print("Creating dataset...")
    dataset = CurveDataset(
        data_path=config['data_path'],
        store_gnames = True,
        init_transform=init_transform,
        get_item_transform=get_item_transform,
        total = 10_000,
    )

    gridname_to_dataset = {
        'default': CurveDatasetSubset(dataset, grid_name='default'),
        'extra': CurveDatasetSubset(dataset, grid_name='extra'),
    }

    return gridname_to_dataset


def check_all_weights_exist(model_params: Iterable, models_root: str) -> None:
    for _, _, w_fname in model_params:
        if not os.path.exists(os.path.join(models_root, w_fname)):
            raise ValueError(f"Path {w_fname} does not exist.")
        

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--num-workers', type=int, default=1)
    p.add_argument('--config', type=str)
    args = p.parse_args()
    return args 



if __name__ == '__main__':
    args = parse_args()
    config = get_config(args.config)

    check_all_weights_exist(config['model_params'], config['models_root'])

    gridname_to_dataset = get_gridname_to_dataset(config)

    for grid_name, model_getter_name, weights_f_name in config['model_params']:

        out_path = os.path.join(config['out_path'],
                                f"{weights_f_name.replace('/', '__')}.pkl")
        
        if os.path.exists(out_path):
            print(f"Path {out_path} exists. Skipping.")
            continue

        predictor = Predictor(
            model_getter_name,
            os.path.join(config['models_root'], weights_f_name),
            config['generator'],
            generator_kwargs=config['generator_kwargs']
        )

        preds_and_meta = predictor.predict(
            gridname_to_dataset[grid_name],
            grid_name, config['data_split'], args.num_workers)

        save_predictions(preds_and_meta, out_path, config["csv_path"])
