from typing import Dict
import json


def get_wh(grid: dict) -> tuple:
    return grid['width'], grid['height']

def get_gname_to_wh(gname_to_grid: Dict[str, dict]):
    return {gname: get_wh(grid)
            for gname, grid in gname_to_grid.items()}

def get_kb_label(key: dict) -> str:
    if 'label' in key:
        return key['label']
    if 'action' in key:
        return key['action']
    raise ValueError("Key has no label or action property")

def get_grid(grid_name: str, grids_path: str) -> dict:
    with open(grids_path, "r", encoding="utf-8") as f:
        return json.load(f)[grid_name]
    
def get_grid_name_to_grid(grid_name_to_grid__path: str, 
                          allowed_gnames = ("default", "extra")) -> dict:
    # In case there will be more grids in "grid_name_to_grid.json"
    grid_name_to_grid = {
        grid_name: get_grid(grid_name, grid_name_to_grid__path)
        for grid_name in allowed_gnames
    }
    return grid_name_to_grid