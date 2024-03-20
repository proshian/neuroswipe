# command:
# python 


from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import argparse

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from tokenizers import CharLevelTokenizerv2
from word_generators import _prepare_encoder_input
from model import MODEL_GETTERS_DICT
from dataset import CurveDataset



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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vocab-path', type=str, help='Path to the vocabulary file')
    p.add_argument('--model-weights-path', type=str, help='Path to the model weigts')
    p.add_argument('--model-name', type=str)
    p.add_argument('--grid-name', type=str, help='Name of the grid')
    p.add_argument('--device', type=str, help='Device to use')
    p.add_argument('--dataset_path', type=str, help='Path to the dataset')
    p.add_argument('--num-workers', type=int, help='Number of workers to use')
    return p.parse_args()


def get_vocab(vocab_path) -> List[str]:
    with open(vocab_path, 'r', encoding = 'utf-8') as f:
        return f.read_lines()

if __name__ == '__main__':
    args = parse_args()
    model = MODEL_GETTERS_DICT[args.model_name](args.device, args.model_weights_path)
    tokenizer = CharLevelTokenizerv2(args.vocab_path)
    vocab = get_vocab(args.vocab_path)
    print(vocab)
