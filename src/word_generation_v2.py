from typing import List, Tuple, Dict
from torch import Tensor
from word_generators import GreedyGenerator
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional

from tqdm import tqdm

from dataset import NeuroSwipeDatasetv2


def process_example(data: Tuple[int, Tuple[Tensor, Tensor, Tensor], str, Dict[str, GreedyGenerator]],
                    generator_kwargs = None) -> Tuple[int, str]:
    if generator_kwargs is None:
        generator_kwargs = {}
    i, gen_in, grid_name, grid_name_to_word_generator = data
    pred = grid_name_to_word_generator[grid_name](*gen_in, **generator_kwargs)
    pred = pred
    return i, pred
    

def predict_raw_mp(dataset: NeuroSwipeDatasetv2,
                   grid_name_to_greedy_generator: dict,
                   num_workers: int=3,
                   generator_kwargs: Optional[dict] = None,
                   ) -> List[List[str]]:
    """
    Creates predictions given a word generator
    
    Arguments:
    ----------
    dataset: NeuroSwipeDatasetv2
        Output[i] is a list of predictions for dataset[i] curve.
    grid_name_to_greedy_generator: dict
        Dict mapping each grid name to a corresponing word generator object.
        A word generator has a model as a property and returns a list of tuples
        (log probability, char_sequence).
    num_workers: int
        Number of processes.
    generator_kwargs: Optional[dict]
        Dictionalry of kwargs of word_generator __init__ method.
        Currently used in BeamGenerator only.
    """
    if generator_kwargs is None:
        generator_kwargs = {}

    preds = [None] * len(dataset)

    data = [(i, (xyt, kb_tokens, traj_pad_mask), grid_name, grid_name_to_greedy_generator)
            for i, ((xyt, kb_tokens, _, traj_pad_mask, _), _, grid_name) in enumerate(dataset)]
    
    process_example_ = partial(process_example, generator_kwargs=generator_kwargs)

    with ProcessPoolExecutor(num_workers) as executor:
        for i, pred in tqdm(executor.map(process_example_, data), total=len(dataset)):
            preds[i] = pred

    return preds