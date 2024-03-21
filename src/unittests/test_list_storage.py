import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "src"))

import unittest
import shutil

import torch

from create_and_save_fully_transformed_ds import BinsListStorage


class TestBinsListStorage(unittest.TestCase):
    def setUp(self) -> None:
        self.bls = BinsListStorage(bin_size=3)
        self.out_path = "test_bins_list_storage_temp_dir"
    
    def test_storage(self):
        data = [(torch.rand((3, 4)), torch.rand(2, 3)) for _ in range(41)]
        self.bls.write(self.out_path, data)
        read_data = self.bls.read(self.out_path)
        self.assertEqual(len(data), len(read_data))
        for i in range(len(data)):
            self.assertTrue(torch.equal(data[i][0], read_data[i][0]))
            self.assertTrue(torch.equal(data[i][1], read_data[i][1]))
    
    def tearDown(self) -> None:
        shutil.rmtree(self.out_path)

    
if __name__ == '__main__':
    unittest.main()