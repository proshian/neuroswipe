from typing import List, Tuple, Set, Dict, Optional
from abc import ABC, abstractmethod
import heapq
from collections import defaultdict

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



class WordGeneratorWithVocab(WordGenerator):
    def __init__(self, model: torch.nn.Module, 
                 tokenizer: CharLevelTokenizerv2, device,
                 vocab: Optional[List[str]] = None,
                 max_token_id = None) -> None:
        """
        Arguments:
        ----------
        vocab: Optional[List[str]]
            List of all possible words.
            It's used to mask out the tokens that can't follow
            generated prefix.
            If vocab is provided, max_token_id must be provided too.
        max_token_id: Optional[int]
            The maximum token id that can be generated.
            A model might never generate some tokens. For example,
            we never need to generate <pad> or <sos> tokens.
            max_token_id == n_out_neurons - 1 == n_classes - 1.
            It's supposed that if model doesn't generate some tokens,
            the unallowed tokens correspond to the last n_tokens - n_out_neurons
            tokens in the tokenizer.
        """

        if max_token_id is None and vocab is not None:
            raise ValueError(
                "If vocab is provided max_token_id must be provided too")
        
        super().__init__(model, tokenizer, device)

        self.max_token_id = max_token_id
        self.vocab = vocab
        self.prefix_to_allowed_chars = None
        if vocab is not None:
            self.prefix_to_allowed_chars = self._create_prefix_to_allowed_tokens(vocab)

    
    def _create_prefix_to_allowed_tokens(self, vocab: List[str]) -> Dict[str, Set[str]]:
        # ! When switching to another type of tokenizer where tokens are not just characters
        # but can be a sequence of characters, we need to change the implementation of this method. 
        prefix_to_allowed_chars = defaultdict(set)
        prefix_to_allowed_chars[''] = set(self.tokenizer.char_to_idx.keys())
        for word in vocab:
            for i in range(1, len(word)):
                prefix = word[:i]
                prefix_to_allowed_chars[prefix].add(word[i])
            prefix_to_allowed_chars[word].add('<eos>')
        return prefix_to_allowed_chars
    
    def _get_unallowed_token_ids(self, prefix: List[str]) -> Set[int]:
        if self.prefix_to_allowed_chars is None:
            return set()
        allowed_chars = self.prefix_to_allowed_chars[prefix]
        all_chars = set(self.tokenizer.char_to_idx.keys())
        impossible_ids = set(range(self.max_token_id + 1, len(self.tokenizer.char_to_idx)))
        impossible_chars = {self.tokenizer.idx_to_char[idx] for idx in impossible_ids}
        unallowed_chars = all_chars - allowed_chars - impossible_chars
        return {self.tokenizer.char_to_idx[char] for char in unallowed_chars}
    
    def _mask_out_unallowed_ids(self, prefix_ids: List[int], logits: Tensor
                                ) -> Tensor:
        if self.prefix_to_allowed_chars is None:
            return logits
        str_prefix__no_sos = self.tokenizer.decode(prefix_ids[1:])
        unallowed_ids = self._get_unallowed_token_ids(str_prefix__no_sos)
        logits[torch.tensor(list(unallowed_ids), dtype = torch.int)] = float('-inf')
        return logits



class GreedyGenerator(WordGeneratorWithVocab):
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
            curve_pad_mask = None

            next_tokens_logits = self.model.decode(
                encoded, dec_in_char_seq, curve_pad_mask, word_pad_mask).transpose_(0, 1)[0, -1]
            next_tokens_logits = self._mask_out_unallowed_ids(tokens, next_tokens_logits)
            next_tokens_logproba = F.log_softmax(next_tokens_logits)
            best_next_token = int(next_tokens_logproba.argmax())  # batch_i = 0, decoder_out_onehot_vector_seq_i = -1 
            log_prob += float(next_tokens_logproba[best_next_token])
            tokens.append(best_next_token)
            if best_next_token == self.eos_token_id:
                break

        return [(log_prob, self.tokenizer.decode(tokens[1:]))]

    def __call__(self, xyt, kb_tokens, max_steps_n=35) -> List[Tuple[float, str]]:
        return self._generate(xyt, kb_tokens, max_steps_n)
    
    def generate_word_only(self, xyt, kb_tokens, max_steps_n=35) -> str:
        return self._generate(xyt, kb_tokens, max_steps_n)[0][1]



class BeamGenerator(WordGeneratorWithVocab):
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

        # Partial hypotheses is a heap (stored as a list) of tuples.
        # Each tuple consists of a partial (unfinished aka intermidiate)
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
            curve_pad_mask = None

            
            next_tokens_logits = self.model.decode(
                encoded, dec_in_char_seq, curve_pad_mask, word_pad_mask).transpose_(0, 1)[0, -1]
            next_tokens_logits = self._mask_out_unallowed_ids(cur_partial_hypothesis, next_tokens_logits)
            next_tokens_logproba = F.log_softmax(next_tokens_logits)
            topk_continuations = next_tokens_logproba.topk(beamsize)

            for token_score, token_idx in zip(topk_continuations.values, topk_continuations.indices):
                # Convert tesors to loat and int to avoid memory leakage.
                token_score = float(token_score)
                token_idx = int(token_idx)

                # Skipping tokens with prob = 0 (log_prob = -inf).
                # Theese tokens apper because even if there's less 
                # then `beamsize` tokens with non-zero probs
                # topk()  will still return exactly `beamsize` tokens. 
                # There are two sourses of zero prob: 
                # 1. Model is extremely confident (maybe overconfident) 
                #    that a certain token is impossible with a given prefix.
                # 2. Masking out unallowed tokens makes their prob = 0.
                if token_score == float('-inf'):
                    continue

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
from tqdm.auto import tqdm


def prepare_words(words: List[str], tokenizer: CharLevelTokenizerv2,
                  batch_first: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    probs: torch.Tensor
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
