import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "src"))

import unittest
import os
import json

from dataset import NeuroSwipeDatasetv2
from tokenizers import KeyboardTokenizerv1, CharLevelTokenizerv2

# проверить, что если "раздеть" decoder_input и target имеют корректную структуру:
# то есть decoder_input[0] = <sos>, дальше пока элементы являются кирилическими
# буквами они совпадают, когда кончились кирилличекие буквы в decoder_input <pad>,
# в decoder output - <eos>, дальше везду pad


def get_grid(grid_name: str, grids_path: str) -> dict:
    with open(grids_path, "r", encoding="utf-8") as f:
        return json.load(f)[grid_name]


class TestDataset(unittest.TestCase):

    def setUp(self) -> None:      
        self.keyboard_selection_set=set(KeyboardTokenizerv1.i2t)

        DATA_ROOT = "data/data_separated_grid"

        grid_name_to_grid_path = os.path.join(DATA_ROOT, "gridname_to_grid.json")
        self.grid_name_to_grid = {grid_name: get_grid(grid_name, grid_name_to_grid_path)
                            for grid_name in ["default", "extra"]}


        val_path = os.path.join(DATA_ROOT, f"valid__in_train_format.jsonl")

        kb_tokenizer = KeyboardTokenizerv1()
        word_tokenizer = CharLevelTokenizerv2(os.path.join(DATA_ROOT, "voc.txt")) 
        
            
        self.val_dataset = NeuroSwipeDatasetv2(
            data_path = val_path,
            gridname_to_grid = self.grid_name_to_grid,
            kb_tokenizer = kb_tokenizer,
            max_traj_len = 299,
            word_tokenizer = word_tokenizer,
            include_time = False,
            include_velocities = True,
            include_accelerations = True,
            has_target=True,
            has_one_grid_only=False,
            include_grid_name=False,
            keyboard_selection_set=self.keyboard_selection_set,
            total = 10_000
        )

    
    def test_keyboard_selection_set(self):
        
        coord_labels = set()

        for grid_name, grid in self.grid_name_to_grid.items():
            coord_to_nearest_label = self.val_dataset._nearest_kb_label_dict[grid_name]
            for x in range(grid['width']):
                for y in range(grid['height']):
                    coord_labels.add(coord_to_nearest_label[x, y])

        # if coord_labels != self.keyboard_selection_set:
        #     print("selection set and nearest labels set are different!")
        #     print(coord_labels)
        #     print(self.keyboard_selection_set)

        for coord_label in coord_labels:
            self.assertIn(coord_label, self.keyboard_selection_set)
        
        

if __name__ == "__main__":
    unittest.main()