import itertools
from collections import defaultdict
from typing import List


def normalize(w):
    return w.strip().lower()

# -----------------------------------------------------------------------------------------------
#                                      Voc

class Voc:
    def __init__(self, words: List[str]):
        self.words_by_first_letter = defaultdict(list)

        for word in sorted(list(words)):
            first_letter = word[0]
            self.words_by_first_letter[first_letter].append(word)

    @staticmethod
    def read(path, voc_size=-1):
        words = []
        with open(path, encoding="utf-8") as f:
            for l in f:
                word = l.rstrip().split('\t')[0]
                if '-' in word:
                    continue
                words.append(normalize(word))

        if voc_size > 0:
            voc = Voc(words[:voc_size])
        else:
            voc = Voc(words)
        return voc

    def get_words_by_first_letter(self, letter: str):
        if letter in self.words_by_first_letter:
            return self.words_by_first_letter[letter]
        
        return list(itertools.chain.from_iterable(self.words_by_first_letter.values()))
