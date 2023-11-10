from typing import List, Tuple, Dict
from torch import Tensor
from word_generators import GreedyGenerator
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time

from tqdm import tqdm

# def dummy_test_predict_example(i_and_data):
#     time.sleep(0.5)
#     i, data = i_and_data
#     return i, data

# def dummy_test_get_model_predictions(dataset):
#     """
#     Creates submission file generating words greedily.

#     If prediction is not in the vocabulary 
#     """
#     NUM_WORKERS = 2
#     predictions = [None] * len(dataset)
        
#     with ProcessPoolExecutor(NUM_WORKERS) as executor:
#         process_function = dummy_test_predict_example
#         for idx, result in tqdm(executor.map(process_function, enumerate(dataset)), total=len(dataset)):
#                 predictions[idx] = result
#     return predictions


# from typing import List


def process_example(data: Tuple[int, Tuple[Tensor, Tensor, Tensor], str, Dict[str, GreedyGenerator]]) -> Tuple[int, str]:
    i, gen_in, grid_name, grid_name_to_word_generator = data
    pred = grid_name_to_word_generator[grid_name](*gen_in)
    pred = pred.removeprefix("<sos>")
    return i, pred

def predict_greedy_raw_multiproc(dataset,
                                 grid_name_to_greedy_generator,
                                 num_workers=2
                                ) -> List[List[str]]:
    """
    Creates predictions using greedy generation.
    
    Arguments:
    ----------
    dataset: NeuroSwipeDatasetv2
    grid_name_to_greedy_generator: dict
        Dict mapping grid names to GreedyGenerator objects.
    """
    preds = [None] * len(dataset)

    data = [(i, (xyt, kb_tokens, traj_pad_mask), grid_name, grid_name_to_greedy_generator)
            for i, ((xyt, kb_tokens, _, traj_pad_mask, _), _, grid_name) in enumerate(dataset)]

    with ProcessPoolExecutor() as executor:
        for i, pred in tqdm(executor.map(process_example, data), total=len(dataset)):
            preds[i] = [pred]

    return preds
    