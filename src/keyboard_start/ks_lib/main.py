import argparse
import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os

from baseline import DtwModel
from data import Dataset
from voc import Voc
from fcom import Fcom


# ===============================================================================================
#                                           Helpers

class ParallelProcessor:
    def __init__(self, num_workers):
        self.num_workers = num_workers

    def process(self, data, process_function, *additional_args):
        results = [None] * len(data)
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor: 
            process_function = partial(process_function, *additional_args)
            for idx, result in tqdm.tqdm(executor.map(process_function, enumerate(data)), total=len(data)):
                results[idx] = result

        return results


def process_example(model, index_example):
    idx, example = index_example
    result = model(example)
    return idx, result


def save_results(results, path):
    df = pd.DataFrame(results)
    df.to_csv(path, header=False, index=False)


# ===============================================================================================
#                                           Baseline

def baseline(args):
    data = Dataset(args.test_path if args.test_path else args.train_path)

    voc = Voc.read(args.voc_path, args.voc_size)
    fcom = Fcom(voc)
    dtw_baseline = DtwModel(fcom)

    parallel_processor = ParallelProcessor(args.num_workers)
    results = parallel_processor.process(data, process_example, dtw_baseline)

    save_results(results, args.output_path)


# ===============================================================================================
#                                           Main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train-path', type=str)
    p.add_argument('--test-path', type=str)

    p.add_argument('--voc-path', type=str)
    p.add_argument('--voc-size', type=int, default=-1)

    p.add_argument('--output-path', type=str)

    p.add_argument('--num-workers', type=int, default=1)

    args = p.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    out_dir_name = os.path.dirname(args.output_path)

    if not os.path.exists(out_dir_name):
        raise ValueError("non-existent output directory: '{out_dir_name}'")

    baseline(args)
