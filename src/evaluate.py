from typing import List, Tuple, Dict
import pickle
import json
import argparse
import os
from dataclasses import asdict

import pandas as pd
from tqdm.auto import tqdm

from predict_v2 import Prediction
from metrics import get_mmr, get_accuracy

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str)
    args = p.parse_args()
    return args

def get_config() -> dict:
    args = parse_args()
    config_path = args.config
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def read_prediction(prediction_path) -> Prediction:
    with open(prediction_path, 'rb') as f:
        prediction = pickle.load(f)
    return prediction


def get_labels_from_ds_path(dataset_path: str, 
                            gnames_to_include: List[str]
                            ) -> List[str]:
    labels = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_data = json.loads(line)
            gname = line_data['curve']['grid_name']
            if gname in gnames_to_include:
                labels.append(line_data['word'])
    return labels


def scored_preds_to_raw_preds(scored_preds: List[List[Tuple[float, str]]]
                              ) -> List[List[str]]:
    preds_only = []
    for scored_preds_for_curve_i in scored_preds:
        preds_only.append([pred for score, pred in scored_preds_for_curve_i])
    return preds_only

def cut_inner_lists_to_four(preds):
    return [preds_for_curve[:4] for preds_for_curve in preds]

def leave_one_pred_per_curve(preds):
    return [preds_for_curve[0] for preds_for_curve in preds]


def is_result_in_df(df: pd.DataFrame, result: dict) -> bool:
    for _, row in df.iterrows():
        for key, value in result.items():
            assert type(row[key]) == type(value), f'{key}: dtype in result is {type(value)}, in df is {type(row[key])}'

    for _, row in df.iterrows():
        if row.to_dict() == result:
            return True
    return False          


def save_results(prediction_with_meta: Prediction, 
                 metrics: Dict[str, float], out_path: str) -> None:
    prediction_with_meta_dict = asdict(prediction_with_meta)
    prediction_with_meta_dict.update(metrics)
    prediction_with_meta_dict['generator_call_kwargs'] = json.dumps(
        prediction_with_meta_dict['generator_call_kwargs'])
    del prediction_with_meta_dict['prediction']

    df_line = pd.DataFrame([prediction_with_meta_dict])
    if not os.path.exists(out_path):
        df_line.to_csv(out_path, index=False)
    else:
        df = pd.read_csv(out_path)
        if not is_result_in_df(df, prediction_with_meta_dict):
            df_line.to_csv(out_path, mode='a', header=False, index=False)


def list_files_recursive(dir_path: str, f_paths: List[str]):
    for entry in os.listdir(dir_path):
        full_path = os.path.join(dir_path, entry)
        if os.path.isdir(full_path):
            list_files_recursive(full_path, f_paths)
        else:
            f_paths.append(full_path)


def list_files_recursive_for_list(all_paths: str, f_paths: List[str]):
    for path in all_paths:
        if os.path.isdir(path):
            all_paths_in_this_dir = []
            list_files_recursive(path, all_paths_in_this_dir)
            f_paths.extend(all_paths_in_this_dir)
        else:
            f_paths.append(path)
    

def get_prediction_paths(config) -> List[str]:
    paths = []
    list_files_recursive_for_list(config['prediction_paths'], paths)
    return paths



def evaluate_path(prediction_path, config) -> None:
    prediction_with_meta = read_prediction(prediction_path)
    preds = scored_preds_to_raw_preds(prediction_with_meta.prediction)
    data_split = prediction_with_meta.dataset_split
    dataset_path = config['data_split__to__path'][data_split]
    labels = get_labels_from_ds_path(dataset_path, 
                                        prediction_with_meta.grid_name)
    mmr = get_mmr(cut_inner_lists_to_four(preds), 
                    labels)
    accuracy = get_accuracy(leave_one_pred_per_curve(preds),
                            labels)
    metrics = {
        'mmr': mmr,
        'accuracy': accuracy
    }

    save_results(prediction_with_meta, metrics, config['out_csv_path'])


if __name__ == "__main__":
    config = get_config()
    prediction_paths = get_prediction_paths(config)
    for prediction_path in tqdm(prediction_paths):
        evaluate_path(prediction_path, config)
