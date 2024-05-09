from typing import Dict

def get_gname_to_wh(gname_to_grid: Dict[str, dict]):
    return {gname: (grid['width'], grid['height']) 
            for gname, grid in gname_to_grid.items()}


def get_kb_label(key: dict) -> str:
    if 'label' in key:
        return key['label']
    if 'action' in key:
        return key['action']
    raise ValueError("Key has no label or action property")