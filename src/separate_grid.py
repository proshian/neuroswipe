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
        print(f"Warning {out_path} already exists. Skipping.")
        return
    
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



def separate_grid_primitive(data_path: str,
                            out_path: str,
                            json_grid_str_to_gridname: Dict[str, str],
                            total: Optional[int]  = None) -> None:
    """
    Faster (around 10x) fragile and dangerous alternative to separate_grid.

    Manipulates json as a string. It works only because grid is the last
    field in the json and the same grid is always exactly the same string.
    It was checked that theese conditions are met for all the .jsonl files
    in ./data/data. It was also checked that
    separate_grid_primitive and separate_grid results are exactly the same.
    However, this is a very fragile solution and thus it should be used
    carefully and only if time is critical.

    json_grid_str_to_gridname = {
        '"grid":{"width":1080,"height":667,"keys":[{"label":"й","hitbox":{"x":0,"y":15,"w":99,"h":154}},{"label":"ц","hitbox":{"x":98,"y":15,"w":99,"h":154}},{"label":"у","hitbox":{"x":196,"y":15,"w":100,"h":154}},{"label":"к","hitbox":{"x":295,"y":15,"w":99,"h":154}},{"label":"е","hitbox":{"x":393,"y":15,"w":99,"h":154}},{"label":"н","hitbox":{"x":491,"y":15,"w":99,"h":154}},{"label":"г","hitbox":{"x":589,"y":15,"w":99,"h":154}},{"label":"ш","hitbox":{"x":687,"y":15,"w":99,"h":154}},{"label":"щ","hitbox":{"x":785,"y":15,"w":100,"h":154}},{"label":"з","hitbox":{"x":884,"y":15,"w":99,"h":154}},{"label":"х","hitbox":{"x":982,"y":15,"w":98,"h":154}},{"label":"ф","hitbox":{"x":0,"y":169,"w":99,"h":154}},{"label":"ы","hitbox":{"x":98,"y":169,"w":99,"h":154}},{"label":"в","hitbox":{"x":196,"y":169,"w":100,"h":154}},{"label":"а","hitbox":{"x":295,"y":169,"w":99,"h":154}},{"label":"п","hitbox":{"x":393,"y":169,"w":99,"h":154}},{"label":"р","hitbox":{"x":491,"y":169,"w":99,"h":154}},{"label":"о","hitbox":{"x":589,"y":169,"w":99,"h":154}},{"label":"л","hitbox":{"x":687,"y":169,"w":99,"h":154}},{"label":"д","hitbox":{"x":785,"y":169,"w":100,"h":154}},{"label":"ж","hitbox":{"x":884,"y":169,"w":99,"h":154}},{"label":"э","hitbox":{"x":982,"y":169,"w":98,"h":154}},{"action":"shift","hitbox":{"x":0,"y":323,"w":120,"h":154}},{"label":"я","hitbox":{"x":119,"y":323,"w":94,"h":154}},{"label":"ч","hitbox":{"x":212,"y":323,"w":95,"h":154}},{"label":"с","hitbox":{"x":306,"y":323,"w":94,"h":154}},{"label":"м","hitbox":{"x":399,"y":323,"w":95,"h":154}},{"label":"и","hitbox":{"x":493,"y":323,"w":94,"h":154}},{"label":"т","hitbox":{"x":586,"y":323,"w":95,"h":154}},{"label":"ь","hitbox":{"x":680,"y":323,"w":94,"h":154}},{"label":"б","hitbox":{"x":773,"y":323,"w":95,"h":154}},{"label":"ю","hitbox":{"x":867,"y":323,"w":95,"h":154}},{"action":"backspace","hitbox":{"x":961,"y":323,"w":119,"h":154}},{"action":"toNumberState","hitbox":{"x":0,"y":477,"w":141,"h":154}},{"action":"globe","hitbox":{"x":140,"y":477,"w":120,"h":154}},{"label":",","hitbox":{"x":259,"y":477,"w":98,"h":154}},{"action":"space","hitbox":{"x":356,"y":477,"w":455,"h":154}},{"label":".","hitbox":{"x":810,"y":477,"w":98,"h":154}},{"action":"enter","hitbox":{"x":907,"y":477,"w":173,"h":154}}],"grid_name":"default"}}}\n' : "default",
        '"grid":{"width":1080,"height":667,"keys":[{"label":"й","hitbox":{"x":0,"y":15,"w":91,"h":154}},{"label":"ц","hitbox":{"x":90,"y":15,"w":91,"h":154}},{"label":"у","hitbox":{"x":180,"y":15,"w":91,"h":154}},{"label":"к","hitbox":{"x":270,"y":15,"w":91,"h":154}},{"label":"е","hitbox":{"x":360,"y":15,"w":91,"h":154}},{"label":"н","hitbox":{"x":450,"y":15,"w":91,"h":154}},{"label":"г","hitbox":{"x":540,"y":15,"w":91,"h":154}},{"label":"ш","hitbox":{"x":630,"y":15,"w":91,"h":154}},{"label":"щ","hitbox":{"x":720,"y":15,"w":91,"h":154}},{"label":"з","hitbox":{"x":810,"y":15,"w":91,"h":154}},{"label":"х","hitbox":{"x":900,"y":15,"w":91,"h":154}},{"label":"ё","hitbox":{"x":990,"y":15,"w":90,"h":154}},{"label":"ф","hitbox":{"x":0,"y":169,"w":91,"h":154}},{"label":"ы","hitbox":{"x":90,"y":169,"w":91,"h":154}},{"label":"в","hitbox":{"x":180,"y":169,"w":91,"h":154}},{"label":"а","hitbox":{"x":270,"y":169,"w":91,"h":154}},{"label":"п","hitbox":{"x":360,"y":169,"w":91,"h":154}},{"label":"р","hitbox":{"x":450,"y":169,"w":91,"h":154}},{"label":"о","hitbox":{"x":540,"y":169,"w":91,"h":154}},{"label":"л","hitbox":{"x":630,"y":169,"w":91,"h":154}},{"label":"д","hitbox":{"x":720,"y":169,"w":91,"h":154}},{"label":"ж","hitbox":{"x":810,"y":169,"w":91,"h":154}},{"label":"э","hitbox":{"x":900,"y":169,"w":91,"h":154}},{"label":"ъ","hitbox":{"x":990,"y":169,"w":90,"h":154}},{"action":"shift","hitbox":{"x":0,"y":323,"w":91,"h":154}},{"label":"я","hitbox":{"x":90,"y":323,"w":91,"h":154}},{"label":"ч","hitbox":{"x":180,"y":323,"w":91,"h":154}},{"label":"с","hitbox":{"x":270,"y":323,"w":91,"h":154}},{"label":"м","hitbox":{"x":360,"y":323,"w":91,"h":154}},{"label":"и","hitbox":{"x":450,"y":323,"w":91,"h":154}},{"label":"т","hitbox":{"x":540,"y":323,"w":91,"h":154}},{"label":"ь","hitbox":{"x":630,"y":323,"w":91,"h":154}},{"label":"б","hitbox":{"x":720,"y":323,"w":91,"h":154}},{"label":"ю","hitbox":{"x":810,"y":323,"w":91,"h":154}},{"label":"?","hitbox":{"x":900,"y":323,"w":91,"h":154}},{"action":"backspace","hitbox":{"x":990,"y":323,"w":90,"h":154}},{"action":"toNumberState","hitbox":{"x":0,"y":477,"w":141,"h":154}},{"action":"globe","hitbox":{"x":140,"y":477,"w":120,"h":154}},{"label":",","hitbox":{"x":259,"y":477,"w":98,"h":154}},{"action":"space","hitbox":{"x":356,"y":477,"w":455,"h":154}},{"label":".","hitbox":{"x":810,"y":477,"w":98,"h":154}},{"action":"enter","hitbox":{"x":907,"y":477,"w":173,"h":154}}],"grid_name":"extra"}}}\n' : "extra"
    }
    """
    with open(data_path, encoding="utf-8") as f, open(out_path, 'a', encoding="utf-8") as out_f:
        for i, line in tqdm(enumerate(f), total = total):

            grid_index = line.find('"grid"')
            gridname = json_grid_str_to_gridname[line[grid_index:]]
            line_gridname_instead_of_grid = line[:grid_index]+f'"grid_name":"{gridname}"\u007d\u007d\n'
            out_f.write(line_gridname_instead_of_grid)



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
    ORIG_ROOT = "data/data"

    f_names = ['train.jsonl', 'valid.jsonl', 'test.jsonl']
    data_paths = [os.path.join(ORIG_ROOT, f_name) for f_name in f_names]
    out_paths = [os.path.join(OUT_ROOT, f_name) for f_name in f_names]
    totals = [6_000_000, 10_000, 10_000]


    create_all_datasets_with_separated_grid(data_paths, out_paths, totals)


    grid_name_to_grid = get_grid_name_to_grid(data_paths[-1], totals[-1])

    grid_name_to_grid__path = os.path.join(OUT_ROOT, "gridname_to_grid.json")

    with open(grid_name_to_grid__path, 'w', encoding='utf-8') as f:
        json.dump(grid_name_to_grid, f, ensure_ascii=False, separators=(',', ':'), indent=2)
        