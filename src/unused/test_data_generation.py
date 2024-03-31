import array
import random

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import pyarrow as pa


ALL_CYRILLIC_LETTERS_ALPHABET_ORD = [
    'а', 'б', 'в', 'г', 'д', 'е', 'ë', 'ж', 'з', 'и', 'й',
    'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф',
    'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я'
]



data_list__trajfeats_kbtokens_tgtword = []

for _ in tqdm(range(1000)):
    curve_len = random.randint(10, 100)
    kb_tokens = torch.randint(0, 127, (curve_len,))
    kb_tokens = array.array('B', kb_tokens)

    traj_feats = np.array(torch.rand((6, curve_len)).to(dtype = torch.float32))

    tgt_word_len = random.randint(4, 15)
    tgt_word = ''.join(random.choices(
        ALL_CYRILLIC_LETTERS_ALPHABET_ORD, k=tgt_word_len))
    
    data = (traj_feats, kb_tokens, tgt_word)

    data_list__trajfeats_kbtokens_tgtword.append(data)
