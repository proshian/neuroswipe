import heapq
import numpy as np
import typing as tp
from tslearn.metrics import dtw as ts_dtw

from fcom import Features, Fcom

# -----------------------------------------------------------------------------------------------
#                                      DtwModel

class DtwModel:
    def __init__(self, fcom: Fcom):
        self.fcom = fcom

    def __call__(self, data: tp.Dict, top_cnt: int = 4) -> tp.List[str]:
        feats: Features = self.fcom(data)
        curve = feats.target_curve

        best_candidates_heap = []

        for (cand, cand_curve) in feats.candidates:
            cur_dtw = ts_dtw(cand_curve, curve)

            if len(best_candidates_heap) < top_cnt:
                heapq.heappush(best_candidates_heap, (-cur_dtw, cand))
            elif cur_dtw < -best_candidates_heap[0][0]:
                heapq.heappushpop(best_candidates_heap, (-cur_dtw, cand))

        return [cand for _, cand in sorted(best_candidates_heap, key=lambda x: -x[0])]
