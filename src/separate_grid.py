from typing import List, Dict, Optional
import os
import json

from tqdm import tqdm


def get_grid_name_to_grid(data_path: str,
                          total: Optional[int] = None
                          ) -> Dict[str, dict]:
    grid_name_to_grid = {}
    with open(data_path, 'r', encoding="utf-8") as f:
        for line in tqdm(f, total = total):
            line_data = json.loads(line)
            grid = line_data['curve']['grid']
            grid_name_to_grid[grid['grid_name']] = grid
    return grid_name_to_grid


def separate_grid(data_path: str,
                  out_path: str,
                  total: Optional[int]  = None) -> None:                                             
    if os.path.exists(out_path):
        raise ValueError(f"File {out_path} already exists!")
    
    with open(data_path, 'r', encoding="utf-8") as f, open(out_path, 'w', encoding="utf-8") as out_f:
        for line in tqdm(f, total = total):
            line_data = json.loads(line)

            g_name = line_data['curve']['grid']['grid_name']

            line_data['curve']['grid_name'] = g_name

            del line_data['curve']['grid']

            json.dump(line_data,
                        out_f,
                        ensure_ascii=False,
                        separators=(',', ':'))
            out_f.write('\n')


def create_all_datasets_with_separated_grid(data_paths: List[str],
                                            out_paths: List[str],
                                            totals: List[Optional[int]]
                                            ) -> None:
    assert len(data_paths) == len(out_paths) == len(totals)

    for data_path, out_path, total in zip(data_paths, out_paths, totals):
        # сделать функцию, генерирующую новую версию одного файла и запустить цикл
        separate_grid(
            data_path,
            out_path,
            total
        )
    
    

if __name__ == '__main__':
    OUT_ROOT = "data/data_separated_grid"
    ORIG_ROOT = "data/data/result_noctx_10k"

    f_names = ['train.jsonl', 'valid.jsonl', 'test.jsonl']
    data_paths = [os.path.join(ORIG_ROOT, f_name) for f_name in f_names]
    out_paths = [os.path.join(OUT_ROOT, f_name) for f_name in f_names]
    totals = [6_000_000, 10_000, 10_000]


    create_all_datasets_with_separated_grid(data_paths, out_paths, totals)


    grid_name_to_grid = get_grid_name_to_grid(data_paths[-1], totals[-1])

    grid_name_to_grid__path = os.path.join(OUT_ROOT, "grid_name_to_grid.json")

    with open(grid_name_to_grid__path, 'w', encoding='utf-8') as f:
        json.dump(grid_name_to_grid, f, ensure_ascii=False, separators=(',', ':'), indent=2)
        