from typing import List, Tuple, Dict
from torch import Tensor
from word_generators import WordGenerator
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional

from tqdm import tqdm

from dataset import NeuroSwipeDatasetv3


I_GenIn_GName_GNameToWordGen = (
    Tuple[int, Tuple[Tensor, Tensor], str, Dict[str, WordGenerator]])

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
                   ) -> List[List[str]]:
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