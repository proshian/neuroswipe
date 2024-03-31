from typing import List, Tuple
from abc import ABC, abstractmethod
import heapq

import torch
import torch.nn.functional as F
from torch import Tensor

from tokenizers import CharLevelTokenizerv2


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
    

##################################################################
# The sectoin below is for VocabEstimator.
# It's unfinished

from concurrent.futures import ProcessPoolExecutor
from functools import partial

from tqdm import tqdm



@torch.inference_mode()
def get_word_prob(model: torch.nn.Module, encoded_curve: Tensor, 
                  tokenizer: CharLevelTokenizerv2, word: str) -> float:
    device = encoded_curve.device
    encoded_word_lst = tokenizer.encode(word)
    decoder_in = torch.tensor(encoded_word_lst[:-1]).reshape(-1, 1).to(device)
    decoder_target = torch.tensor(encoded_word_lst[1:])
    word_pad_mask = None
    logits = model.decode(encoded_curve, decoder_in, None, word_pad_mask).transpose_(0, 1)[0]
    logproba = F.log_softmax(logits, dim=1)
    logprob = sum(logproba[range(len(decoder_target)), decoder_target])
    return logprob.item()



@torch.inference_mode()
def get_vocab_probs(model: torch.nn.Module, vocab: List[str], 
                    xyt: Tensor, kb_tokens: Tensor, device: str,
                    batch_first: bool, tokenizer: CharLevelTokenizerv2,
                    num_workers: int = 4
                    ) -> List[float]:
    """
    For each word from `vocab` predicts the likelihood of the word
    given the curve represented by `xyt` and `kb_tokens`. 
    The likelihood is calculated as a sum of log-probabilities of
    each token in the word.
    """
    model.to(device)
    xyt, kb_tokens = _prepare_encoder_input(
        xyt, kb_tokens, device, batch_first)
    encoded_curve = model.encode(xyt, kb_tokens, None)
    logprobs = []
    prob_getter = partial(get_word_prob, model, encoded_curve, tokenizer)
    with ProcessPoolExecutor(num_workers) as executor:
        for logprob in tqdm(executor.map(prob_getter, vocab), total=len(vocab)):
            logprobs.append(logprob)



class VocabEstimator(WordGenerator):
    def __init__(self, model, tokenizer, device, vocab_path):
        super().__init__(model, tokenizer, device)
        self.vocab = self._get_vocab(vocab_path)
    
    def _get_vocab(vocab_path) -> List[str]:
        with open(vocab_path, 'r', encoding = 'utf-8') as f:
            return f.read().splitlines()
    
    @torch.inference_mode()
    def __call__(self, xyt, kb_tokens, return_hypotheses_n=None) -> List[Tuple[float, str]]:
        return_hypotheses_n = return_hypotheses_n or len(self.vocab)

        xyt, kb_tokens = _prepare_encoder_input(xyt, kb_tokens,
                                                self.device, False)
        
        encoded = self.model.encode(xyt, kb_tokens, None)



##################################################################



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
            # dummy_y is any tensor with n_dims = 2 (chars_seq_len - 1, batch_size).
            dummy_y = torch.tensor([[1]])
            model_input = (xyt, kb_tokens, dec_in_char_seq, None, word_pad_mask)
            # model_input, dummy_y = prepare_batch(model_input, dummy_y, self.device)
            one_hot_token_logits = self.model.apply(*model_input).transpose_(0, 1)  # (batch_size, chars_seq_len, vocab_size)
            best_next_tokens = one_hot_token_logits[:, -1].argmax(dim=1)  # (batch_size)

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