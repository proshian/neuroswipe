from typing import Tuple, List, Set, Dict
import os
import pickle
import copy
import json


def remove_beamsearch_probs(preds: List[List[Tuple[float, str]]]) -> List[List[str]]:
    new_preds = []
    for pred_line in preds:
        new_preds_line = []
        for _, word in pred_line:
            new_preds_line.append(word)
        new_preds.append(new_preds_line)
    return new_preds


def patch_wrong_prediction_shape(prediciton):
    return [pred_el[0] for pred_el in prediciton]


def separate_invalid_preds(preds: List[List[str]],
                           vocab_set: Set[str]
                           ) -> Tuple[List[List[str]], Dict[int, List[str]]]:

    all_real_word_preds = []
    all_errorous_word_preds = {}

    for i, pred in enumerate(preds):
        real_word_preds = []
        errorous_word_preds = []
        for word in pred:
            if word in vocab_set:
                real_word_preds.append(word)
                if len(real_word_preds) == 4:
                    break
            else:
                errorous_word_preds.append(word)
        
        all_real_word_preds.append(real_word_preds)
        if len(real_word_preds) < 4:
            all_errorous_word_preds[i] = errorous_word_preds

    return all_real_word_preds, all_errorous_word_preds


def augment_predictions(preds:List[List[str]], augment_list: List[List[str]]):
    augmented_preds = copy.deepcopy(preds)
    for pred_line, aug_l_line in zip(augmented_preds, augment_list):
        for aug_el in aug_l_line:
            if len(pred_line) >= 4:
                break
            if not aug_el in pred_line:
                pred_line.append(aug_el)
    return augmented_preds


def merge_preds(default_preds,
                extra_preds,
                default_idxs,
                extra_idxs):
    preds = [None] * (len(default_preds) + len(extra_preds))

    for i, val in zip(default_idxs, default_preds):
        preds[i] = copy.deepcopy(val)
    for i, val in zip(extra_idxs, extra_preds):
        preds[i] = copy.deepcopy(val)

    return preds


def get_vocab_set(vocab_path: str):
    with open(vocab_path, 'r', encoding = "utf-8") as f:
        return set(f.read().splitlines())


def get_default_and_extra_idxs(test_dataset_path
                               ) -> Tuple[List[int], List[int]]:
    default_idxs = []
    extra_idx = []
    with open(test_dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line_data = json.loads(line)
            grid_name = line_data['curve']['grid_name']
            if grid_name == 'default':
                default_idxs.append(i)
            elif grid_name == 'extra':
                extra_idx.append(i)
            else:
                raise ValueError(f"Unexpected grid_name: {grid_name}.")
    
    return default_idxs, extra_idx


def create_submission(preds_list, out_path) -> None:
    if os.path.exists(out_path):
        raise ValueError(f"File {out_path} already exists")
    
    with open(out_path, "w", encoding="utf-8") as f:
        for preds in preds_list:
            pred_str = ",".join(preds)
            f.write(pred_str + "\n")
            


if __name__ == "__main__":
    DATA_ROOT = "data/data_separated_grid/"

    grid_name_to_ranged_bs_model_preds_paths = {
        'default': [
            "m1_bigger__m1_bigger_v2__2023_11_12__14_51_49__0.13115__greed_acc_0.86034__default_l2_0_ls0_switch_2.pt.pkl",
            "m1_bigger__m1_bigger_v2__2023_11_12__12_30_29__0.13121__greed_acc_0.86098__default_l2_0_ls0_switch_2.pt.pkl",
            "m1_bigger__m1_bigger_v2__2023_11_11__22_18_35__0.13542_default_l2_0_ls0_switch_1.pt.pkl",
            "m1_v2__m1_v2__2023_11_09__10_36_02__0.14229_default_switch_0.pt.pkl",
            "m1_bigger__m1_bigger_v2__2023_11_12__00_39_33__0.13297_default_l2_0_ls0_switch_1.pt.pkl",
            "m1_bigger__m1_bigger_v2__2023_11_11__14_29_37__0.13679_default_l2_0_ls0_switch_0.pt.pkl",
            
        ],
        'extra': [
            "m1_v2__m1_v2__2023_11_09__17_47_40__0.14301_extra_l2_1e-05_switch_0.pt.pkl",
            "m1_bigger__m1_bigger_v2__2023_11_12__02_27_14__0.13413_extra_l2_0_ls0_switch_1.pt.pkl"
        ]
    }

    vocab_set = get_vocab_set(os.path.join(DATA_ROOT, "voc.txt"))

    default_idxs, extra_idxs = get_default_and_extra_idxs(
        os.path.join(DATA_ROOT, "test.jsonl"))

    grid_name_to_augmented_preds = {}

    for grid_name in ('default', 'extra'):
        bs_pred_list = []

        for f_name in grid_name_to_ranged_bs_model_preds_paths[grid_name]:
            f_path = os.path.join("data/saved_beamsearch_results/", f_name)
            with open(f_path, 'rb') as f:
                bs_pred_list.append(pickle.load(f))
            
        bs_pred_list = [patch_wrong_prediction_shape(bs_preds) for bs_preds in bs_pred_list] 
        bs_pred_list = [remove_beamsearch_probs(bs_preds) for bs_preds in bs_pred_list]
        bs_pred_list = [separate_invalid_preds(bs_preds, vocab_set)[0] for bs_preds in bs_pred_list]


        augmented_preds = bs_pred_list.pop(0)

        while bs_pred_list:
            augmented_preds = augment_predictions(augmented_preds, bs_pred_list.pop(0))

        grid_name_to_augmented_preds[grid_name] = augmented_preds


    full_preds = merge_preds(
        grid_name_to_augmented_preds['default'],
        grid_name_to_augmented_preds['extra'],
        default_idxs,
        extra_idxs)
    

    baseline_preds = None
    with open(r"data\submissions\baseline.csv", 'r', encoding = 'utf-8') as f:
        baseline_preds = f.read().splitlines()
    baseline_preds = [line.split(",") for line in baseline_preds]

    full_preds = augment_predictions(full_preds, baseline_preds)

    create_submission(full_preds,
        f"data/submissions/id3_with_baseline_without_old_preds.csv")