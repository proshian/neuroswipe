from typing import List, Tuple, Set, Dict
import heapq

import editdistance


def get_real_and_errorous_words(preds: List[List[Tuple[float, str]]],
                                vocab_set: Set[str]) -> Tuple[List[List[str]], Dict[int, List[str]]]:
    """
    Arguments:
    ----------
    preds: List[List[Tuple[float, str]]]
        preds[i] stores raw output of a wordgenerator corresponding
        to the i-th curve. The raw output is a list of tuples, where
        each tuple is (-log_probability, word). The list is sorted
        by -log_probability in ascending order.
    vocab_set: Set[str]
        A set of all possible words.

    Returns:
    --------
    all_real_word_preds: List[List[str]]
        all_real_word_preds stores 4 lists of real words sorted by
        -log_probability in ascending order.
    all_errorous_word_preds: Dict[int, List[str]]
        all_errorous_word_preds[i] stores a list of errorous words
        sorted by -log_probability in ascending order if all_real_word_preds[i]
        has less than 4 words. Otherwise, all_errorous_word_preds does not
        have the key i.
    """

    all_real_word_preds = []
    all_errorous_word_preds = {}

    for i, pred in enumerate(preds):
        real_word_preds = []
        errorous_word_preds = []
        for _, word in pred:
            if word in vocab_set:
                real_word_preds.append(word)
                if len(real_word_preds) == 4:
                    break
            else:
                errorous_word_preds.append(word)
        
        all_real_word_preds.append(real_word_preds)
        if len(real_word_preds) < 4:
            all_errorous_word_preds[i] = errorous_word_preds

    return all_real_word_preds, all_errorous_word_preds



class MinEditDistance:
    def __init__(self, vocab_set: Set[str]) -> None:
        self.vocab_set = vocab_set
        self.word_to_min_edit_distance_word: Dict[str, Tuple[int, str]] = {}

    def save_state(self, path: str) -> None:
        with open(path, 'w') as f:
            for word, (min_edit_distance, min_edit_distance_word) in self.word_to_min_edit_distance_word.items():
                f.write(f'{word}\t{min_edit_distance}\t{min_edit_distance_word}\n')

    def load_state(self, path: str) -> None:
        with open(path, 'r') as f:
            for line in f:
                word, min_edit_distance, min_edit_distance_word = line.strip().split('\t')
                self.word_to_min_edit_distance_word[word] = (int(min_edit_distance), min_edit_distance_word)

    def get_min_edit_distance_words(self, word: str) -> Tuple[int, List[str]]:
        """
        For a given word return all words in vocab_set with the minimum edit distance and the minimum edit distance.
        """
        if word in self.vocab_set:
            return (0, [word])
        
        if word in self.word_to_min_edit_distance_word:
            return self.word_to_min_edit_distance_word[word]
        

        min_edit_distance = float('inf')
        min_edit_distance_words = []
        for vocab_word in self.vocab_set:
            edit_distance = editdistance.eval(word, vocab_word)
            if edit_distance < min_edit_distance:
                min_edit_distance = edit_distance
                min_edit_distance_words = [vocab_word]
            elif edit_distance == min_edit_distance:
                min_edit_distance_words.append(vocab_word)

        self.word_to_min_edit_distance_word[word] = (min_edit_distance, min_edit_distance_words)
        return (min_edit_distance, min_edit_distance_words)


def augment_real_word_with_min_edit_distance(real_word_preds: List[str],
                                             errorous_word_preds: List[str],
                                             vocab_set: Set[str],
                                             med_getter: MinEditDistance) -> List[str]:
    """
    Given real_words and errorous_words for one curve, augment real_words with
    words with minimum edit distance to errorous_words.
    
    Arguments:
    ----------
    real_word_preds: List[str]
        A list of real words sorted by -log_probability in ascending order.
    errorous_word_preds: List[str]
        A list of errorous words sorted by -log_probability in ascending order.
    vocab_set: Set[str]
        A set of all possible words.

    Returns:
    --------
    augmented_real_word_preds: List[str]
        A list of real words sorted by -log_probability in ascending order.
        The list is augmented with words with minimum edit distance to the real words.
    """

    if len(real_word_preds) == 4:
        return real_word_preds
    
    n_with_med_1 = 0
    aug_candidates = []
    
    for errorous_word in errorous_word_preds:
        min_edit_distance, min_edit_distance_words = med_getter.get_min_edit_distance_words(errorous_word)
        for min_edit_distance_word in min_edit_distance_words:
            if min_edit_distance_word not in real_word_preds:
                heapq.heappush(aug_candidates, (min_edit_distance, min_edit_distance_word))
                if min_edit_distance == 1:
                    n_with_med_1 += 1
                if n_with_med_1 >= 4 - len(real_word_preds):
                    break

    while len(real_word_preds) < 4 and len(aug_candidates) > 0:
        _, word = heapq.heappop(aug_candidates)
        real_word_preds.append(word)

    return real_word_preds



def augment_all_real_words_with_min_edit_distance(all_real_word_preds: List[List[str]],
                                                  all_errorous_word_preds: Dict[int, List[str]],
                                                    vocab_set: Set[str],
                                                    med_getter: MinEditDistance) -> List[List[str]]:
    
    for i in all_errorous_word_preds:
        all_real_word_preds[i] = augment_real_word_with_min_edit_distance(all_real_word_preds[i],
                                                                          all_errorous_word_preds[i],
                                                                          vocab_set,
                                                                          med_getter)
    return all_real_word_preds



def raw_preds_to_augmented(preds: List[List[Tuple[float, str]]],
                           vocab_set: Set[str],
                           med_getter: MinEditDistance) -> List[List[str]]:
    
    all_real_word_preds, all_errorous_word_preds = get_real_and_errorous_words(preds, vocab_set)
    augmented_preds = augment_all_real_words_with_min_edit_distance(
        all_real_word_preds,
        all_errorous_word_preds,
        vocab_set,
        med_getter)
    return augmented_preds