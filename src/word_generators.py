from typing import List, Tuple
from abc import ABC, abstractmethod
import heapq

import torch
import torch.nn.functional as F
from torch import Tensor

from ns_tokenizers import CharLevelTokenizerv2


def _prepare_encoder_input(xyt: Tensor, kb_tokens: Tensor, 
                            device: str, batch_first: bool
                            ) -> Tuple[Tensor, Tensor]:
    xyt, kb_tokens = (el.unsqueeze(0) for el in (xyt, kb_tokens))
    xyt, kb_tokens = (el.to(device) for el in (xyt, kb_tokens))
    if not batch_first:
        xyt, kb_tokens = (el.transpose(0, 1) for el in (xyt, kb_tokens))
    return xyt, kb_tokens


class WordGenerator(ABC):
    def __init__(self, model: torch.nn.Module, 
                 tokenizer: CharLevelTokenizerv2, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.eos_token_id = tokenizer.char_to_idx['<eos>']

    @abstractmethod
    def __call__(self, xyt, kb_tokens, max_steps_n, 
                 *args, **kwargs) -> List[Tuple[float, str]]:
        pass


class GreedyGenerator(WordGenerator):
    @torch.inference_mode()
    def _generate(self, xyt, kb_tokens, max_steps_n=35) -> List[Tuple[float, str]]:
        tokens = [self.tokenizer.char_to_idx['<sos>']]
        log_prob = 0
        
        xyt, kb_tokens = _prepare_encoder_input(xyt, kb_tokens, 
                                                self.device, False)

        encoded = self.model.encode(xyt, kb_tokens, None)

        for _ in range(max_steps_n):
            
            dec_in_char_seq = torch.tensor(tokens).reshape(-1, 1).to(self.device)  # (chars_seq_len, batch_size)
            # word_pad_mask = torch.zeros_like(dec_in_char_seq, dtype=torch.bool, device=self.device).transpose_(0,1)
            word_pad_mask = None

            next_tokens_logits = self.model.decode(encoded, dec_in_char_seq, None, word_pad_mask).transpose_(0, 1)[0, -1]
            best_next_token = int(next_tokens_logits.argmax())  # batch_i = 0, decoder_out_onehot_vector_seq_i = -1 
            log_prob += float(F.log_softmax(next_tokens_logits, dim=0)[best_next_token])
            if best_next_token == self.eos_token_id:
                break

            tokens.append(int(best_next_token))
    
        return [(log_prob, self.tokenizer.decode(tokens[1:]))]

    def __call__(self, xyt, kb_tokens, max_steps_n=35) -> List[Tuple[float, str]]:
        return self._generate(xyt, kb_tokens, max_steps_n)
    
    def generate_word_only(self, xyt, kb_tokens, max_steps_n=35) -> str:
        return self._generate(xyt, kb_tokens, max_steps_n)[0][1]



class BeamGenerator(WordGenerator):
    @torch.inference_mode()
    def __call__(self,
                 xyt, kb_tokens,
                 max_steps_n=35,  # max tokens in a seq
                 return_hypotheses_n=4,  # n best hypothesis to return
                 beamsize=6,  # n best solutions we store in intermidiate comuptations
                 normalization_factor=0.5,
                 verbose=False
                 ) -> List[Tuple[float, str]]:
        tokens = [self.tokenizer.char_to_idx['<sos>']]
        initial_length = len(tokens)

        # Partial hypothesis is a heap (stored as a list) of tuples.
        # Each tuple consists of a partial (unfinishedaka intermidiate)
        # hypothesis and it's weight.
        # Weight is a measure of likelihood of the hypothesis.
        # [(w1, hypothesis1), (w2, hypothesis2), ...] 
        partial_hypotheses = [(0, tokens)]
        final_hypotheses = []


        xyt, kb_tokens = _prepare_encoder_input(xyt, kb_tokens, 
                                                self.device, False)

        encoded = self.model.encode(xyt, kb_tokens, None)

        while len(partial_hypotheses) > 0:
            cur_partial_score, cur_partial_hypothesis = heapq.heappop(partial_hypotheses)


            dec_in_char_seq = torch.tensor(cur_partial_hypothesis).reshape(-1, 1).to(self.device)  # (chars_seq_len, batch_size)
            # word_pad_mask = torch.zeros_like(dec_in_char_seq, dtype=torch.bool, device=self.device).transpose_(0,1)
            word_pad_mask = None

            
            next_tokens_logits = self.model.decode(encoded, dec_in_char_seq, None, word_pad_mask).transpose_(0, 1)[0, -1]
            next_tokens_logproba = F.log_softmax(next_tokens_logits)
            topk_continuations = next_tokens_logproba.topk(beamsize)

            for token_score, token_idx in zip(topk_continuations.values, topk_continuations.indices):
                # Convert tesors to loat and int to avoid memory leakage.
                token_score = float(token_score)
                token_idx = int(token_idx)

                # score - нормализованная разность log_softmax всех токенов.
                # Разность, а не сумма, потому что heapq - мин-куча. 
                old_denorm_score = cur_partial_score * len(cur_partial_hypothesis)**normalization_factor
                new_score = (old_denorm_score - token_score) / (len(cur_partial_hypothesis) + 1)**normalization_factor

                new_hypothesis = cur_partial_hypothesis + [token_idx]
                new_item = (new_score, new_hypothesis)

                if token_idx == self.eos_token_id or len(new_hypothesis) - initial_length >= max_steps_n:
                    final_hypotheses.append(new_item)
                else:
                    heapq.heappush(partial_hypotheses, new_item)

            if len(partial_hypotheses) > beamsize:
                partial_hypotheses = heapq.nsmallest(beamsize, partial_hypotheses)
                heapq.heapify(partial_hypotheses)

        final_scores, final_token_lists = zip(*final_hypotheses)
        final_texts = [self.tokenizer.decode(final_token_list[1:-1]) for final_token_list in final_token_lists]
        result = list(zip(final_scores, final_texts))
        result.sort()

        if verbose:
            print(result)

        return result[:return_hypotheses_n]




# The class below is not finished. It's suposed that we process
# multiple curves simpultaniously. At each step we check 
# if all batch out sequences have <eos> token. If true,
# generation is finished 
class GreedyGeneratorBatched(WordGenerator):    
    @torch.inference_mode()
    def __call__(self,
                 xyt,  # (batch_size, curves_seq_len, n_coord_feats)
                 kb_tokens,  # (batch_size, curves_seq_len)
                 max_steps_n=35):
        batch_size, curves_seq_len, n_coord_feats = xyt.shape

        # (batch_size, chars_seq_len)
        dec_in_char_seq = torch.full((batch_size, 1), self.tokenizer.char_to_idx['<sos>'], dtype=torch.int, device=self.device)

        # We don't have to put everything to device because it's done in prepare_batch.

        for _ in range(max_steps_n):
            word_pad_mask = None
            model_input = (xyt, kb_tokens, dec_in_char_seq, None, word_pad_mask)
            # model_input = prepare_batch(model_input, self.device)
            logits = self.model.apply(*model_input).transpose_(0, 1)  # (batch_size, chars_seq_len, vocab_size)
            best_next_tokens = logits[:, -1].argmax(dim=1)  # (batch_size)

            print(best_next_tokens.shape, dec_in_char_seq.shape)

            dec_in_char_seq = torch.cat((dec_in_char_seq, best_next_tokens.unsqueeze(0)), dim=0)
            kb_tokens.transpose_(0, 1)
            xyt.transpose_(0,1)

        predictions = [self.tokenizer.decode(dec_in_char_seq[i].tolist()) for i in range(batch_size)]
        
        return predictions




GENERATOR_CTORS_DICT = {
    "greedy": GreedyGenerator,
    "beam": BeamGenerator
}














############################################################
###############       Vocab Estimation       ###############
############################################################


from torch.utils.data import DataLoader  # for typing 
from torch.nn.utils.rnn import pad_sequence





def prepare_words(words: List[str], tokenizer: CharLevelTokenizerv2,
                  batch_first: bool, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    word_pad_idx = tokenizer.char_to_idx['<pad>']
    
    word_ids = [torch.tensor(tokenizer.encode(word), dtype=torch.int) for word in words]
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
#                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch_first:
        n_words = word_ids.shape[0]
        word_ids = word_ids.expand(n_words * batch_size, -1)
    else:
        n_words = word_ids.shape[1]
        word_ids = word_ids.expand(-1, n_words * batch_size)
    
    word_ids = word_ids.to(device)

    return word_ids
    


@torch.inference_mode()
def estimate_probs_of_words(model: torch.nn.Module, dataloader: DataLoader, 
                            words: List[str], tokenizer: CharLevelTokenizerv2, 
                            batch_first, device: str) -> torch.Tensor:
    """
    For each curve in the dataloader estimates the probability of each word in the words list.

    Arguments:
    ----------
    model: torch.nn.Module
        The model that is used to estimate the probability of the words.
    dataloader: DataLoader (uses datasets.CurveDataset)
        contains the curves for which we want to estimate 
        the probability of the words.
    words:
        The list of possible words.
    tokenizer: CharLevelTokenizerv2
        The tokenizer that encodes the words.
    batch_first: bool
        If True, the batch is the first dimention of the input.
    device: str
        The device on which the model is stored

    Returns:
    --------
    probs: torch.Tensor
        The tensor of shape (batch_size, n_words) where
        probs[i, j] is the probability of the word j for the curve i.


    Algorithm:
    ----------
    0) Prepare the words: tokenize, pad, create mask and expand dims to (seq_len, batch_size*n_words)
    1) Encode batch of curves
    2) Expand encoded_curves' shape to (seq_len, batch_size*n_words, emb_size)

    """

    n_words = len(words)


    model.eval()
    model.to(device)

    batch_size = dataloader.batch_size

    word_ids_in, word_ids_out, word_pad_mask = prepare_words(
        words, tokenizer, batch_first, batch_size)
    
    word_ids_in, expand_word_ids(
        word_ids_in, batch_size, batch_first, device)
    
    word_pad_mask = word_pad_mask.expand(n_words * batch_size, -1)
    
    batch_results = []

    for batch in dataloader:
        batch_x, _ = batch
        traj_feats, kb_embs, _, curve_pad_mask, _ = batch_x

        traj_feats, kb_embs, curve_pad_mask = (
            el.to(device) for el in (traj_feats, kb_embs, curve_pad_mask))

        x_encoded = model.encode(traj_feats, kb_embs, curve_pad_mask)  # (d_model, batch_size, seq_len)


        # need to expand x_encoded for each word in the batch
        x_encoded = x_encoded.expand(-1, batch_size * n_words, -1)
        x_encoded = x_encoded.to(device)
        curve_pad_mask.expand(batch_size * n_words, -1)

        logits = model.decode(x_encoded, word_ids_in, curve_pad_mask, word_pad_mask)

        if batch_first:
            logits = logits.transpose(0, 1)
        
        # logits.shape = (batch_size*n_words, seq_len, vocab_size)
        
        _, char_seq_len, vocab_size = logits.shape 

        log_probs = F.log_softmax(logits, dim=-1)  # ! check dim
        log_probs_zeroed_pad = log_probs.masked_fill(word_pad_mask.unsqueeze(-1), 0)  
        log_probs_zeroed_pad = log_probs_zeroed_pad.reshape(batch_size, n_words, char_seq_len, vocab_size)
        # log_probs_zeroed_pad.shape = (batch_size, n_words, max_seq_len, vocab_size)

        # let's for each curve i and word j let's collect the probability of the word j

        # thus will get a tensor of size (batch_size, n_words, max_seq_len)



        log_probs_of_words = log_probs_zeroed_pad.gather(3, word_ids_out.unsqueeze(-1)).squeeze(-1)
        # log_probs_of_words.shape = (batch_size, n_words, max_seq_len)

        # let's sum the log_probs_of_words over the last dimention
        # to get the probability of the word

        log_probs_of_words = log_probs_of_words.sum(dim=-1)  # (batch_size, n_words)

        batch_results.append(log_probs_of_words)

    return torch.cat(batch_results, dim=0)

