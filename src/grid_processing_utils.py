from typing import Dict
import json


def get_gname_to_wh(gname_to_grid: Dict[str, dict]):
    return {gname: (grid['width'], grid['height']) 
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