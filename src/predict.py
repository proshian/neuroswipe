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


# Если предсказания были получены для валидационного датасета, хочется уметь
# Считать метрики. 


# Есть желание получить абстракцию. В частности абстракцию сохранения данных.
# Создать класс Predictor, который при инициализации создает поле с word_generator, 
# model_id, ...
# Имеет метод predict, метод save_predictions



from typing import Callable, List, Tuple, Dict, Optional
import os
import json
import pickle
import argparse
from functools import partial, partialmethod

import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

from model import get_m1_model, get_m1_bigger_model, get_m1_smaller_model
from tokenizers import CharLevelTokenizerv2
from dataset import NeuroSwipeDatasetv3, NeuroSwipeGridSubset
from tokenizers import KeyboardTokenizerv1
from word_generators import WordGenerator, BeamGenerator, GreedyGenerator


MODEL_GETTERS_DICT = {
    "m1": get_m1_model,
    "m1_bigger": get_m1_bigger_model,
    "m1_smaller": get_m1_smaller_model
}

GENERATOR_CTORS_DICT = {
    "greedy": GreedyGenerator,
    "beam": BeamGenerator
}

I_GenIn_GName_GNameToWordGen = (
    Tuple[int, Tuple[Tensor, Tensor], str, Dict[str, WordGenerator]])


class Predictor:
    """
    Creates a prediction for a whole dataset.

    Predictor is similar to WordGenerator, but WordGenerator is
    just an abstract decoding algorithm (that hides the model),
    while a Predictor does not hide anything.  Its instance is
    associated with model AND GRID so that it can be 
    reffered to by an aggregator as a certain source of predictions.
    """

    def __init__(self,
                 model_architecture_name,
                 model_weights_path,
                 grid_name,
                 word_generator_type,
                 word_char_tokenizer,
                 out_path,
                 preds_csv_path,
                 dataset_split,
                 generator_kwargs=None,
                 ) -> None:
        DEVICE = torch.device('cpu')

        self._word_generator_type = word_generator_type
        word_generator_ctor = GENERATOR_CTORS_DICT[word_generator_type]
        self.model_architecture_name = model_architecture_name
        self.model_weights_path = model_weights_path
        model_getter = MODEL_GETTERS_DICT[model_architecture_name]
        model = model_getter(DEVICE, model_weights_path)
        self.grid_name = grid_name
        self.word_char_tokenizer = word_char_tokenizer
        self.word_generator = word_generator_ctor(model, word_char_tokenizer, DEVICE)
        self.generator_kwargs = generator_kwargs
        self.out_path = out_path
        self.preds_csv_path = preds_csv_path
        self.dataset_split = dataset_split

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
    
    def _predict_raw_mp(self, dataset: NeuroSwipeDatasetv3,
                        num_workers: int=3) -> List[List[str]]:
        """
        Creates predictions given a word generator
        
        Arguments:
        ----------
        dataset: NeuroSwipeDatasetv3
            The dataset is supposed to be a subset of the original dataset
            containing only examples with the same grid_name as the predictor.
        num_workers: int
            Number of processes.
        """
        preds = [None] * len(dataset)

        data = [(i, (traj_feats, kb_tokens))
                for i, ((traj_feats, kb_tokens, _, _), _) in enumerate(dataset)]
        

        with ProcessPoolExecutor(num_workers) as executor:
            for i, pred in tqdm(executor.map(self._predict_example, data), total=len(dataset)):
                preds[i] = pred

        return preds

    def predict(self, dataset: NeuroSwipeDatasetv3,
                num_workers: int=3) -> List[List[Tuple[float, str]]]:
        """
        Creates predictions given a word generator
        
        Arguments:
        ----------
        dataset: NeuroSwipeDatasetv3
            Output[i] is a list of predictions for dataset[i] curve.
        num_workers: int
            Number of processes.
        """
        preds = self._predict_raw_mp(dataset, num_workers)
        self.preds = preds
        return preds
    
    def get_id(self):
        """
        Given a csv table get the predictor_id. 
        The predictor_id is a string that uniquely identifies the predictor.
        It is defined by generator_type, generator_call_kwargs_json,
        model_architecture_name, model_weights_path, grid_name.

        
        The predictions_table.csv has the following columns:
        * predictor_id
        * word_generator_type
        * Generator_call_kwargs_json
        * Model_architecture_name
        * Model_weights_path
        * Grid_name
        * test_preds_path
        * val_preds_path
        * validation_metric
        """
        
        # have to find the row with same 
        # word_generator_type, word_generator_call_kwargs_json,
        # model_architecture_name, model_weights_path, grid_name
        # and return the predictor_id from that row

        df = pd.read_csv(self.preds_csv_path)
        df_line = df.loc[df['word_generator_type'] == self._word_generator_type & 
                    df['word_generator_call_kwargs_json'] == json.dumps(self.generator_kwargs) &
                    df['model_architecture_name'] == self.model.__class__.__name__ &
                    df['model_weights_path'] == self.model.weights_path &
                    df['grid_name'] == self.grid_name]
        
        assert len(df_line) < 2, "There may be one or zero rows with same predictor"
        
        if len(df) == 0:
            # create a new row
            df = df.append({
                'word_generator_type': self._word_generator_type,
                'word_generator_call_kwargs_json': json.dumps(self.generator_kwargs),
                'model_architecture_name': self.model.__class__.__name__,
                'model_weights_path': self.model.weights_path,
                'grid_name': self.grid_name,
                'test_preds_path': None,
                'val_preds_path': None,
                'validation_metric': None
            }, ignore_index=True)
            df.to_csv(self.preds_csv_path, index=False)

        return df_line['predictor_id'].values[0]


    def save_predictions(self):
        with open(self.out_path, 'wb') as f:
            pickle.dump(self.preds, f, protocol=pickle.HIGHEST_PROTOCOL)

        predictor_id = self.get_id()
        df = pd.read_csv(self.preds_csv_path)
        df.loc[df['predictor_id'] == predictor_id, f'{self.dataset_split}_preds_path'] = self.out_path
        df.to_csv(self.preds_csv_path, index=False)

        

                         




def predict_example(data: I_GenIn_GName_GNameToWordGen,
                    generator_kwargs: dict = None
                    ) -> Tuple[int, List[Tuple[float, str]]]:
    """
    Predicts a single example.

    Arguments:
    ----------
    data: I_GenIn_GName_GNameToWordGen
        Tuple of (i, gen_in, grid_name, grid_name_to_word_generator).
        i: int
            Index of the example in the dataset.
        gen_in: Tuple[Tensor, Tensor]
            Tuple of (traj_feats, kb_tokens).
        grid_name: str
            Name of the grid.
        grid_name_to_word_generator: Dict[str, WordGenerator]
            Dict mapping each grid name to a corresponing word generator object.
            A word generator has a model as a property and returns a list of tuples
            (log probability, char_sequence).
    generator_kwargs: Optional[dict]
        Dictionalry of kwargs of word_generator __call__ method.
        Currently used in BeamGenerator only, GreedyGenerator
        takes an empty dict.

    Returns:
    --------
    i: int
        Index of the example in the dataset.
    pred: List[Tuple[float, str]]
        List of tuples (log_probability, char_sequence).
    """
    if generator_kwargs is None:
        generator_kwargs = {}
    i, gen_in, grid_name, grid_name_to_word_generator = data
    pred = grid_name_to_word_generator[grid_name](*gen_in, **generator_kwargs)
    pred = pred
    return i, pred
    

def predict_raw_mp(dataset: NeuroSwipeDatasetv3,
                   grid_name_to_greedy_generator: dict,
                   num_workers: int=3,
                   generator_kwargs: Optional[dict] = None,
                   ) ->  List[List[Tuple[float, str]]]:
    """
    Creates predictions given a word generator
    
    Arguments:
    ----------
    dataset: NeuroSwipeDatasetv3
        Output[i] is a list of predictions for dataset[i] curve.
    grid_name_to_greedy_generator: dict
        Dict mapping each grid name to a corresponing word generator object.
        A word generator has a model as a property and returns a list of tuples
        (log_probability, char_sequence).
    num_workers: int
        Number of processes.
    generator_kwargs: Optional[dict]
        Dictionalry of kwargs of word_generator __init__ method.
        Currently used in BeamGenerator only.
    """
    if generator_kwargs is None:
        generator_kwargs = {}

    preds = [None] * len(dataset)

    data = [(i, (traj_feats, kb_tokens), grid_name, grid_name_to_greedy_generator)
            for i, ((traj_feats, kb_tokens, _, _), _, grid_name) in enumerate(dataset)]
    
    process_example_ = partial(predict_example, generator_kwargs=generator_kwargs)

    with ProcessPoolExecutor(num_workers) as executor:
        for i, pred in tqdm(executor.map(process_example_, data), total=len(dataset)):
            preds[i] = pred

    return preds



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


def get_grid_name_to_grid(grid_name_to_grid__path: str) -> dict:
    # In case there will be more grids in "grid_name_to_grid.json"
    grid_name_to_grid = {
        grid_name: get_grid(grid_name, grid_name_to_grid__path)
        for grid_name in ("default", "extra")
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


def get_gridname_to_dataset(config,
                            word_char_tokenizer) -> Dict[str, Dataset]:

    kb_tokenizer = KeyboardTokenizerv1()
    keyboard_selection_set = set(kb_tokenizer.i2t)
                                 
    grid_name_to_grid = get_grid_name_to_grid(
        config['grid_name_to_grid__path'])
    
    print("Loading dataset...")
    dataset = NeuroSwipeDatasetv3(
        data_path = config['data_path'],
        gridname_to_grid = grid_name_to_grid,
        kb_tokenizer = kb_tokenizer,
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

    gridname_to_dataset = {
        'default': NeuroSwipeGridSubset(dataset, grid_name='default'),
        'extra': NeuroSwipeGridSubset(dataset, grid_name='extra'),
    }

    return gridname_to_dataset


def check_all_weights_exist(model_params, models_root) -> None:
    for _, _, w_fname in model_params:
        if not os.path.exists(os.path.join(models_root, w_fname)):
            raise ValueError(f"Path {w_fname} does not exist.")
        


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--num-workers', type=int, default=1)
    p.add_argument('--config', type=str)
    args = p.parse_args()
    return args 


def old_main():
    args = parse_args()
    config = get_config(args.config)

    check_all_weights_exist(config['model_params'])

    word_char_tokenizer = CharLevelTokenizerv2(
        config['voc_path'])

    gridname_to_dataset = get_gridname_to_dataset(
        config, word_char_tokenizer)

    for grid_name, model_getter_name, weights_f_name in config['model_params']:

        out_path = os.path.join(config['out_path'],
                                f"{weights_f_name.replace('/', '__')}.pkl")
        
        if os.path.exists(out_path):
            print(f"Path {out_path} exists. Skipping.")
            continue

        predictions = weights_to_raw_predictions(
            grid_name = grid_name,
            model_getter=MODEL_GETTERS_DICT[model_getter_name],
            weights_path = os.path.join(config['models_root'], weights_f_name),
            word_char_tokenizer=word_char_tokenizer,
            dataset=gridname_to_dataset[grid_name],
            generator_ctor=GENERATOR_CTORS_DICT[config['generator']],
            n_workers=args.num_workers,
            generator_kwargs=config['generator_kwargs']
        )

        with open(out_path, 'wb') as f:
            pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args.config)

    check_all_weights_exist(config['model_params'], config['models_root'])

    # ! Мне кажется, с целью абстракции predictor'а есть смысл
    # создавать токенизатор в нем. Ну и в get_gridname_to_dataset
    # тоже перекочует word_char_tokenizer
    word_char_tokenizer = CharLevelTokenizerv2(
        config['voc_path'])

    gridname_to_dataset = get_gridname_to_dataset(
        config, word_char_tokenizer)

    for grid_name, model_getter_name, weights_f_name in config['model_params']:

        out_path = os.path.join(config['out_path'],
                                f"{weights_f_name.replace('/', '__')}.pkl")
        
        if os.path.exists(out_path):
            print(f"Path {out_path} exists. Skipping.")
            continue

        predictor = Predictor(
            model_getter_name,
            os.path.join(config['models_root'], weights_f_name),
            grid_name,
            config['generator'],
            word_char_tokenizer,
            out_path,
            preds_csv_path=None,
            dataset_split=config['data_split'],
            generator_kwargs=config['generator_kwargs']
        )

        predictor.predict(gridname_to_dataset[grid_name], args.num_workers)

        predictor.save_predictions()