import typing as tp
import numpy as np
from dataclasses import dataclass

from voc import Voc
from grid import Grid


@dataclass
class Features:
    target_curve: np.array
    candidates: tp.List[tp.Tuple[str, np.ndarray]]


# -----------------------------------------------------------------------------------------------
#                                      Fcom

class Fcom:
    def __init__(self, voc: Voc):
        self.voc = voc
        self.grids: tp.Dict[str, Grid] = {}
        self.cur_grid: tp.Optional[Grid] = None

    def __call__(self, x: tp.Dict) -> Features:
        self._set_grid(x)

        curve = self._get_curve(x)

        first_letter = self._get_first_letter(curve)
        candidates = self.voc.get_words_by_first_letter(first_letter)
        candidates_curves = [self._get_centered_curve_by_letters(cand) for cand in candidates]

        return Features(
            target_curve=curve,
            candidates=list(zip(candidates, candidates_curves))
        )

    def _get_centered_curve_by_letters(self, word: str) -> np.ndarray:
        return self.cur_grid.get_centered_curve(word)

    def _get_first_letter(self, curve: np.ndarray) -> str:
        return self.cur_grid.get_the_nearest_hitbox(curve[0][0], curve[0][1])

    def _set_grid(self, data: tp.Dict) -> None:
        grid_info = data['curve']['grid']
        grid_name = grid_info['grid_name']

        if grid_name not in self.grids:
            self.grids[grid_name] = Grid(grid_info)

        self.cur_grid = self.grids[grid_name]

    @staticmethod
    def _get_curve(data: tp.Dict) -> np.ndarray:
        curve_data = data['curve']
        xs = curve_data['x']
        ys = curve_data['y']
        return np.array(list(zip(xs, ys)))
