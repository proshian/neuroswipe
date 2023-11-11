from typing import List, Tuple, Dict
from torch import Tensor
from word_generators import GreedyGenerator
from concurrent.futures import ProcessPoolExecutor
from functools import partial


from tqdm import tqdm


def process_example(data: Tuple[int, Tuple[Tensor, Tensor, Tensor], str, Dict[str, GreedyGenerator]],
                    generator_kwargs = None) -> Tuple[int, str]:
    if generator_kwargs is None:
        generator_kwargs = {}
    i, gen_in, grid_name, grid_name_to_word_generator = data
    pred = grid_name_to_word_generator[grid_name](*gen_in, **generator_kwargs)
    pred = pred
    return i, pred


# def predict_greedy_raw_multiproc(dataset,
#                                  grid_name_to_greedy_generator,
#                                  num_workers=3,
#                                 ) -> List[List[str]]:
#     """
#     Creates predictions using greedy generation.
    
#     Arguments:
#     ----------
#     dataset: NeuroSwipeDatasetv2
#     grid_name_to_greedy_generator: dict
#         Dict mapping grid names to GreedyGenerator objects.
#     """
#     preds = [None] * len(dataset)

#     data = [(i, (xyt, kb_tokens, traj_pad_mask), grid_name, grid_name_to_greedy_generator)
#             for i, ((xyt, kb_tokens, _, traj_pad_mask, _), _, grid_name) in enumerate(dataset)]

#     with ProcessPoolExecutor(num_workers) as executor:
#         for i, pred in tqdm(executor.map(process_example, data), total=len(dataset)):
#             preds[i] = [pred]

#     return preds
    

def predict_greedy_raw(dataset,
                       grid_name_to_greedy_generator,
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

    for data in tqdm(enumerate(dataset), total=len(dataset)):
        i, ((xyt, kb_tokens, _, traj_pad_mask, _), _, grid_name) = data
        pred = grid_name_to_greedy_generator[grid_name](xyt, kb_tokens, traj_pad_mask)
        pred = pred.removeprefix("<sos>")
        preds[i] = [pred]

    return preds


def predict_raw_mp(dataset,
                   grid_name_to_greedy_generator,
                   num_workers=3,
                   generator_kwargs = None,
                   ) -> List[List[str]]:
    """
    Creates predictions using greedy generation.
    
    Arguments:
    ----------
    dataset: NeuroSwipeDatasetv2
    grid_name_to_greedy_generator: dict
        Dict mapping grid names to GreedyGenerator objects.
    """
    if generator_kwargs is None:
        generator_kwargs = {}

    preds = [None] * len(dataset)

    data = [(i, (xyt, kb_tokens, traj_pad_mask), grid_name, grid_name_to_greedy_generator)
            for i, ((xyt, kb_tokens, _, traj_pad_mask, _), _, grid_name) in enumerate(dataset)]
    
    process_example_ = partial(process_example, generator_kwargs=generator_kwargs)

    with ProcessPoolExecutor(num_workers) as executor:
        for i, pred in tqdm(executor.map(process_example_, data), total=len(dataset)):
            preds[i] = [pred]

    return preds