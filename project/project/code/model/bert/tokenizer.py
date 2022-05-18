import torch
import collections

class Tokenizer():
    def __init__(self, vocab_file):
        self.vocab = self.load_vocab(vocab_file)

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab

    def __call__(self, splits):
        tokens = {}
        # 动态pad，提取最大长度
        split_length = []
        for split in splits:
            split_length.append(len(split))
        max_length = max(split_length)

        # pad和生成mask，这里默认'[PAD]'=0
        B = len(splits)
        W = max_length
        input_ids = torch.zeros(B, W, dtype=int)
        mask = torch.zeros(B, W, dtype=int)
        for i, split in enumerate(splits):
            for j, word in enumerate(split):
                input_ids[i, j] = self.vocab[word]
                mask[i, j] = 1
                
        tokens['input_ids'] = input_ids
        tokens['attention_mask'] = mask
        
        return tokens
    