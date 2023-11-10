import multiprocessing

# def double(a):
#     return a * 2

# def driver_func():
#     PROCESSES = 4
#     with multiprocessing.Pool(PROCESSES) as pool:
#         params = [(1, ), (2, ), (3, ), (4, )]
#         results = [pool.apply_async(double, p) for p in params]

#         for r in results:
#             print('\t', r.get())


from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time

from tqdm import tqdm

# def predict_example(data, grid_name_to_greedy_generator):
#     i, ((xyt, kb_tokens, _, traj_pad_mask, _), _, grid_name) = data
#     generator = grid_name_to_greedy_generator[grid_name]
#     pred = generator(xyt, kb_tokens, traj_pad_mask)
#     pred = pred.removeprefix("<sos>")
#     return i, pred

def predict_example(i_and_data):
    time.sleep(0.5)
    i, data = i_and_data
    return i, data

def get_model_predictions():
    """
    Creates submission file generating words greedily.

    If prediction is not in the vocabulary 
    """
    NUM_WORKERS = 2
    dataset = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    predictions = [None] * len(dataset)
        
    with ProcessPoolExecutor(NUM_WORKERS) as executor:
        process_function = predict_example
        for idx, result in tqdm(executor.map(process_function, enumerate(dataset)), total=len(dataset)):
                predictions[idx] = result
    return predictions

    

# def get_model_predictions(dataset,
#                           grid_name_to_greedy_generator,
#                           num_workers=2):
#     """
#     Creates submission file generating words greedily.

#     If prediction is not in the vocabulary 
#     """
#     predictions = [None] * len(dataset)
    
#     g2gg = grid_name_to_greedy_generator
    
#     with ProcessPoolExecutor(num_workers) as executor:
#         process_function = partial(predict_example, grid_name_to_greedy_generator=g2gg)
#         for idx, result in tqdm.tqdm(executor.map(process_function, enumerate(dataset)), total=len(dataset)):
#                 predictions[idx] = result
    
    # return predictions

if __name__ == "__main__":

    print(get_model_predictions())