import torch
from torch.utils.data import Dataset
from collections import Counter

class TextDataset(Dataset):
    def __init__(self, lines, word2idx, max_len):
        self.data = []
        self.max_len = max_len
        self.word2idx = word2idx
        for line in lines:
            tokens = [word2idx.get(w, word2idx['<UNK>']) for w in line]
            if len(tokens) >= max_len:
                tokens = tokens[:max_len]
            else:
                tokens += [word2idx['<PAD>']] * (max_len - len(tokens))
            self.data.append(torch.tensor(tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def build_vocab(sentences, min_freq=1):
    counter = Counter(word for sent in sentences for word in sent)
    vocab = ['<PAD>', '<UNK>'] + [word for word, freq in counter.items() if freq >= min_freq]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word

def tokenize_lines(filepath):
    with open(filepath, encoding='utf-8') as f:
        return [line.strip().lower().split() for line in f if line.strip()]
