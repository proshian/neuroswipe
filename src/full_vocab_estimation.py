"""
Implements estimate_probs_of_words function that 
for each curve in the dataloader estimates the probability of each word from
a corresponding word-candidate list.
"""


from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor  # for typing
from torch.utils.data import DataLoader  # for typing 
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from ns_tokenizers import CharLevelTokenizerv2  # for typing


def prepare_words(words: List[str], tokenizer: CharLevelTokenizerv2,
                  batch_first: bool) -> Tuple[Tensor, Tensor, Tensor]:
    word_pad_idx = tokenizer.char_to_idx['<pad>']
    
    word_ids = [Tensor(tokenizer.encode(word), dtype=torch.int) for word in words]
    words_ids_in_no_pad = [word_id[:-1] for word_id in word_ids]
    words_ids_out_no_pad = [word_id[1:] for word_id in word_ids]

    word_ids_in = pad_sequence(words_ids_in_no_pad, batch_first=batch_first, 
                               padding_value=word_pad_idx)
    word_ids_out = pad_sequence(words_ids_out_no_pad, batch_first=batch_first, 
                                padding_value=word_pad_idx)
    
    word_pad_mask = word_ids_out == word_pad_idx

    if not batch_first:
        word_pad_mask = word_pad_mask.T  # word_pad_mask is always batch first

    return word_ids_in, word_ids_out, word_pad_mask


# def expand_word_ids_all(word_ids_in, word_ids_out, word_pad_mask, 
#                     batch_size, batch_first, device,
#                     ) -> Tuple[Tensor, Tensor, Tensor]:
#     if batch_first:
#         n_words = word_ids_in.shape[0]
#         word_ids_in = word_ids_in.expand(n_words * batch_size, -1)
#         word_ids_out = word_ids_out.expand(n_words * batch_size, -1)
#     else:
#         n_words = word_ids_in.shape[1]
#         word_ids_in = word_ids_in.expand(-1, n_words * batch_size)
#         word_ids_out = word_ids_out.expand(-1, n_words * batch_size)
#     word_pad_mask = word_pad_mask.expand(n_words * batch_size, -1)

#     expanded = (word_ids_in, word_ids_out, word_pad_mask)
#     expanded = (el.to(device) for el in expanded)

#     return expanded


def expand_word_ids(word_ids, batch_size, batch_first, device,
                    ) -> Tuple[Tensor, Tensor, Tensor]:
    if batch_first:
        # n_words, seq_len = word_ids.shape
        # word_ids = word_ids.expand(n_words * batch_size, seq_len)
        word_ids = word_ids.repeat(batch_size, 1)
    else:
        # seq_len, n_words = word_ids.shape
        # word_ids = word_ids.expand(seq_len, n_words * batch_size)  
        word_ids = word_ids.repeat(1, batch_size)  
    
    word_ids = word_ids.to(device)

    return word_ids
    


@torch.inference_mode()
def estimate_probs_of_words(model: torch.nn.Module, curve_loader: DataLoader,
                            word_lsts_loader: DataLoader,
                            tokenizer: CharLevelTokenizerv2, 
                            batch_first, device: str) -> Tensor:
    """
    For each curve in the dataloader estimates the probability of each word from
    a corresponding word-candidate list.

    Arguments:
    ----------
    model: torch.nn.Module
        The model that is used to estimate the probability of the words.
    curve_loader: DataLoader (uses datasets.CurveDataset)
        Contains the curves for which we want to estimate 
        the probability of the words.
    word_lsts_loader: DataLoader
        Dataloader based on a torch Dataset such that word_lsts_dataset[i]
        is a list of word candidates for curve curves_dataset[i]
    words: List[List[str]]
        words[i] is the list of all word candidates for dataloader.dataset[i].
    tokenizer: CharLevelTokenizerv2
        The tokenizer that encodes the words.
    batch_first: bool
        If True, the batch is the first dimention of the input.
    device: str
        The device on which the model is stored

    Returns:
    --------
    probs: Tensor
        The tensor of shape (batch_size, n_words) where
        probs[i, j] is the probability of the word j for the curve i.

    Algorithm:
    ----------
    0) Prepare the words: tokenize, pad, create mask and expand dims to (seq_len, batch_size*n_words)
    1) Encode batch of curves
    2) Expand encoded_curves' shape to (seq_len, batch_size*n_words, emb_size)
    """

    # ! Would be more elegant to create a single dataset thet
    #   encapsulates curve data and word candidates.

    # ! May be a good idea to create a tensor of all possible words 
    # ! ( not vocabulary; rather list(set(flatten_list(words))) ).
    # ! For rach curve possible words would be views (slices) of this tensor.

    assert len(curve_loader.dataset) == len(word_lsts_loader.dataset), (
        "The number of curves and the number of word candidate lists must be the same.")
    
    assert curve_loader.batch_size == word_lsts_loader.batch_size, (
        "The batch sizes of curve_loader and word_lsts_loader must be the same.")


    model.eval()
    model.to(device)

    batch_size = curve_loader.batch_size
    
    
    
    
    batch_results = []

    for curve_data, word_candidates in tqdm(zip(curve_loader, word_lsts_loader), total=len(curve_loader)):
        batch_x, _ = curve_data
        traj_feats, kb_embs, _, curve_pad_mask, _ = batch_x

        n_words = len(word_candidates)

        word_ids_in, word_ids_out, word_pad_mask = prepare_words(
            word_candidates, tokenizer, batch_first)
        
        word_ids_in = expand_word_ids(
            word_ids_in, batch_size, batch_first, device)
    
        if not batch_first:
            word_ids_out = word_ids_out.transpose(0, 1)  # (batch_size, seq_len)
        
        # word_pad_mask = word_pad_mask.expand(n_words * batch_size, -1)
        word_pad_mask = word_pad_mask.repeat(batch_size, 1)

        word_ids_in, word_ids_out, word_pad_mask = (
            el.to(device) for el in (word_ids_in, word_ids_out, word_pad_mask))




        traj_feats, kb_embs, curve_pad_mask = (
            el.to(device) for el in (traj_feats, kb_embs, curve_pad_mask))

        x_encoded = model.encode(traj_feats, kb_embs, curve_pad_mask)  # (d_model, batch_size, seq_len)


        # need to expand x_encoded for each word in the batch
        
        if batch_first:
            # x_encoded = x_encoded.expand(batch_size * n_words, -1, -1)
            x_encoded = x_encoded.repeat(n_words, 1, 1)
        else:
            # x_encoded = x_encoded.expand(-1, batch_size * n_words, -1)
            x_encoded = x_encoded.repeat(1, n_words, 1)
            
        x_encoded = x_encoded.to(device)
        # curve_pad_mask = curve_pad_mask.expand(batch_size * n_words, -1)
        curve_pad_mask = curve_pad_mask.repeat(n_words, 1)

        # print(f"x_encoded.shape = {x_encoded.shape}")
        # print(f"word_ids_in.shape = {word_ids_in.shape}")
        # print(f"curve_pad_mask.shape = {curve_pad_mask.shape}")
        # print(f"word_pad_mask.shape = {word_pad_mask.shape}")
        logits = model.decode(x_encoded, word_ids_in, curve_pad_mask, word_pad_mask)


        if not batch_first:
            logits = logits.transpose(0, 1)
        
        _, char_seq_len, vocab_size = logits.shape 
        

        # logits.shape = (batch_size*n_words, seq_len, vocab_size)
        
        # print(f"{char_seq_len = }")

        # log_probs.shape = (char_seq_len, batch_size*n_words, vocab_size)
        # word_pad_mask.shape = (batch_size*n_words, char_seq_len)
        log_probs = F.log_softmax(logits, dim=-1)  # ! check dim
        log_probs_zeroed_pad = log_probs.masked_fill(word_pad_mask.unsqueeze(-1), 0)
        log_probs_zeroed_pad = log_probs_zeroed_pad.reshape(batch_size, n_words, char_seq_len, vocab_size)

        # log_probs_zeroed_pad.shape = (batch_size, n_words, max_seq_len, vocab_size)

        # let's for each curve i and word j let's collect the probability of the word j

        # thus will get a tensor of size (batch_size, n_words, max_seq_len)

        # print(f"{log_probs_zeroed_pad.shape = }")
        # print(f"{word_ids_out.shape = }")

        word_ids_out_expanded = word_ids_out.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 1)

        # print(word_ids_out_expanded.shape)

        # One problem is that neural network never generates <pad> token.
        # I see two solutions:
        # 1) replace each <pad> id occurace in word_ids_out with anything 
        #    else (since we zeroed all log_probs where true label is <pad>).
        # 2) concatenate a tensor of zeros so that 
        #    log_probs_zeroed_pad.shape = (batch_size, n_words, max_seq_len, vocab_size+1)
        #    and log_probs_zeroed_pad[:, :, :, -1] = 0. 
        #    PS. It's supposed that tokenizer.t2i['<pad>'] == vocab_size.
        #
        # Both solutions are not elegant and may be confusing.
        # I will choose the first one. 
        # Maybe it's better to make the token_id swapping before the loop 
        # in the very beginning of this function.

        

        token_to_replace_pad_with = tokenizer.char_to_idx['а']
        word_ids_out_expanded.masked_fill_(
            word_ids_out_expanded == tokenizer.char_to_idx['<pad>'], token_to_replace_pad_with)

        word_ids_out_expanded = word_ids_out_expanded.to(dtype=torch.int64)

        log_probs_of_words = log_probs_zeroed_pad.gather(3, word_ids_out_expanded).squeeze(-1)
        
        # assert log_probs_of_words.shape == (batch_size, n_words, char_seq_len)


        # let's sum the log_probs_of_words over the last dimention
        # to get the probability of the word

        log_probs_of_words = log_probs_of_words.sum(dim=-1)  # (batch_size, n_words)

        # assert log_probs_of_words.shape == (batch_size, n_words)

        batch_results.append(log_probs_of_words.to('cpu'))

    return torch.cat(batch_results, dim=0)
