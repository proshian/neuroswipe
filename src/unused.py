import json

from torch.utils.data import IterableDataset
import torch

class NeuroSwipeIterableDatasetv1(IterableDataset):
    """
    Dataset class for NeuroSwipe dataset.
    The dataset file weights over 3 GB and contains over 6 million swipe gestures.


    """

    def __init__(self, data_path):
        """
        Args:
            data_path (string): Path to the NeuroSwipe dataset in JSON format.
                A custom version of the dataset is used:
                "grid" property is replaced with "grid_name" property.
        """
        self.json_file = open(data_path, "r", encoding="utf-8")

    def __del__(self):
        self.json_file.close()

    def _get_data_from_json_line(self, line):
        """
        Parses a JSON line and returns a dictionary with data.
        """
        data = json.loads(line)
        word: str = data['word']

        X_list = data['curve']['x']
        Y_list = data['curve']['y']
        T_list = data['curve']['t']

        X = torch.tensor(X_list, dtype=torch.float32)
        Y = torch.tensor(Y_list, dtype=torch.float32)
        T = torch.tensor(T_list, dtype=torch.float32)

        return X, Y, T, word
    
    def __iter__(self):
        for line in self.json_file:
            yield self._get_data_from_json_line(line)