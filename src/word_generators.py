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
        self.prefix_ids_to_allowed_token_ids = None
        if vocab is not None:
            self.prefix_to_allowed_ids = self._create_prefix_ids_to_allowed_token_ids(vocab)

    
    def _create_prefix_ids_to_allowed_token_ids(self, vocab: List[str]
                                                ) -> Dict[Tuple[int, ...], Set[int]]:
        # ! When switching to another type of tokenizer where tokens are not just characters
        # but can be a sequence of characters, we need to change the implementation of this method. 
        prefix_to_allowed_ids = defaultdict(set)

        for word in vocab:
            tokenized_word = self.tokenizer.encode(word)
            for i in range(1, len(tokenized_word)):
                prefix = tuple(tokenized_word[:i])
                prefix_to_allowed_ids[prefix].add(tokenized_word[i])
        return prefix_to_allowed_ids
    
    def _get_unallowed_token_ids(self, prefix_ids: List[int]) -> Set[int]:
        if self.prefix_to_allowed_ids is None:
            return set()        
        
        allowed_ids = self.prefix_to_allowed_ids[tuple(prefix_ids)]
        all_ids = set(self.tokenizer.idx_to_char.keys())
        impossible_ids = set(range(self.max_token_id + 1, len(self.tokenizer.char_to_idx)))
        unallowed_ids = all_ids - allowed_ids - impossible_ids

        # print([self.tokenizer.idx_to_char[idx] for idx in prefix_ids])
        # print([self.tokenizer.idx_to_char[idx] for idx in allowed_ids])
        # print([self.tokenizer.idx_to_char[idx] for idx in all_ids])
        # print([self.tokenizer.idx_to_char[idx] for idx in impossible_ids])
        # print([self.tokenizer.idx_to_char[idx] for idx in unallowed_ids])

        return unallowed_ids
    
    def _mask_out_unallowed_ids(self, prefix_ids: List[int], logits: Tensor
                                ) -> Tensor:
        if self.prefix_to_allowed_ids is None:
            return logits
        unallowed_ids = self._get_unallowed_token_ids(prefix_ids)
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
                 return_hypotheses_n: Optional[int] = None,  # n best hypothesis to return
                 beamsize=6,  # n best solutions we store in intermidiate comuptations
                 normalization_factor=0.5,
                 verbose=False
                 ) -> List[Tuple[float, str]]:
        """
        Arguments:
        ----------
        return_hypotheses_n: Optional[int]
            Число возвращаемых гипотез. Если None, возвращаются все,
            иначе возвращается `return_hypotheses_n` наиболее вероятных.
        """
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

        return result if return_hypotheses_n is None else result[:return_hypotheses_n]




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
