import json
from typing import Dict

from fcom import Fcom


# -----------------------------------------------------------------------------------------------
#                                      Dataset

class Dataset:
    def __init__(self, path: str):
        self.data = []

        with open(path, 'r', encoding="utf-8") as file:
            for line in file:
                json_content = json.loads(line)
                self.data.append(json_content)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]
