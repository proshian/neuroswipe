from typing import Collection, List

import torch
from aggregate_predictions import delete_duplicates_stable

def get_mmr(preds_list: Collection[Collection[str]], ref: Collection[str]) -> float:
    # Works properly if has duplicates or n_line_preds < 4

    MMR = 0
    
    for preds, target in zip(preds_list, ref):
        preds = delete_duplicates_stable(preds)
        weights = [1, 0.1, 0.09, 0.08]

        line_MRR = sum(weight * (pred == target)
                       for weight, pred in zip(weights, preds))

        MMR += line_MRR
    
    MMR /= len(preds_list)

    return MMR


def get_accuracy(preds_list: List[str], ref: List[str]) -> float:
    n_equal = sum([int(pred==target) for pred, target in zip(preds_list, ref)])
    return n_equal / len(preds_list)


def get_word_level_accuracy(y_true_batch: torch.Tensor, 
                            pred_batch: torch.Tensor, 
                            pad_token: int, 
                            mask: torch.Tensor) -> float:
    # By default y_true.shape = pred.shape = (chars_seq_len, batch_size)
    # So we have to transpose here or before calling

    y_true_batch = y_true_batch.masked_fill(mask, pad_token)
    pred_batch = pred_batch.masked_fill(mask, pad_token)
    equality_results = torch.all(torch.eq(y_true_batch, pred_batch), dim = 1)
        
    return float(equality_results.sum() / len(equality_results))


def decode_batch(seq_batch, tokenizer):
    return [tokenizer.decode(seq) for seq in seq_batch]


def get_word_level_metric(metric_fn,
                          y_true_batch: torch.Tensor, 
                          pred_batch: torch.Tensor, 
                          char_tokenizer,
                          mask: torch.Tensor) -> float:
    
    y_true_batch.masked_fill_(mask, char_tokenizer.char_to_idx['<pad>'])
    pred_batch.masked_fill_(mask, char_tokenizer.char_to_idx['<pad>'])
        
    y_true_batch = decode_batch(y_true_batch, char_tokenizer)
    pred_batch = decode_batch(pred_batch, char_tokenizer)
    
    return metric_fn(y_true_batch, pred_batch)
