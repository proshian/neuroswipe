from typing import List


class CharLevelTokenizerv2:
    """
    Tokenizes a word into a list of integers.

    Toknized word is padded to the max_word_len.

    Guarantees that <sos> and <pad> are tokens with `vocab_len - 1` and
    `vocab_len - 2` indices respectively. The model never needs to 
    predict <sos> and <pad> tokens. Since theese tokens have the biggest ids
    neuron_index is equal to token_id. Otherwise we would need a mapping
    from neuron_index to token_id
    """
    def __init__(self, vocab_path):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.max_word_len = None  # is set in _build_vocab
        # ! I don't think we will need <unk>, but it
        # doesn't lead to any problems. Decided to keep it.
        self.special_tokens = ["<eos>", "<unk>", "<pad>", "<sos>"]
        self._build_vocab(vocab_path)

    def _build_vocab(self, vocab_path):
        self.max_word_len = 0
        unique_chars = set()

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = f.read().split("\n")
            for word in vocab:
                self.max_word_len = max(self.max_word_len, len(word) + 2)  # + <sos> and <eos>
                for char in word:
                    unique_chars.add(char)
                    
        unique_chars_list = sorted(list(unique_chars)) + self.special_tokens
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars_list)}
        self.idx_to_char = {idx: char for idx, char in enumerate(unique_chars_list)}

    def encode(self, word: str) -> List[int]:
        """
        Tokenizes a word into a list of integers.
        """
        tokenized_word = []
        tokenized_word.append(self.char_to_idx["<sos>"])
        for char in word:
            default: int = self.char_to_idx['<unk>']
            tokenized_word.append(self.char_to_idx.get(char, default))
        tokenized_word.append(self.char_to_idx["<eos>"])
        return tokenized_word
    
    def decode(self, token_seq):
        """
        Decodes a tokenized word into a string.
        """
        return "".join([self.idx_to_char[int(idx)] for idx in token_seq])


class KeyboardTokenizerv1:
    
    i2t = ['а', 'б', 'в', 'г', 'д', 'е', 'ë', 'ж', 'з', 'и', 'й',
           'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф',
           'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я',
           '-', '<unk>', '<pad>']
    
    t2i = {t: i for i, t in enumerate(i2t)}

    def get_token(self, char):
        return self.t2i.get(char, self.t2i['<unk>'])