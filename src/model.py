import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, hidden_dim=64, max_len=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden

    def sample(self, batch_size, device):
        samples = torch.zeros(batch_size, self.max_len, dtype=torch.long).to(device)
        h, c = None, None
        input = torch.zeros(batch_size, 1, dtype=torch.long).to(device)
        for t in range(self.max_len):
            logits, (h, c) = self.forward(input, (h, c) if h is not None else None)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            input = torch.multinomial(probs, 1)
            samples[:, t] = input.view(-1)
        return samples


class Discriminator(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, num_filters=64, filter_sizes=(2, 3, 4), max_len=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, emb_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)

    def forward(self, x):
        embedded = self.embedding(x).unsqueeze(1)
        convs = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        out = torch.cat(pools, dim=1)
        return torch.sigmoid(self.fc(out))
