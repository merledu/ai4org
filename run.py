import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from src.utils import tokenize_lines, build_vocab, TextDataset
from src.model import Generator, Discriminator
from src.rollout import Rollout
from src.train import pretrain_generator, adversarial_train
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 10
batch_size = 64
num_epochs = 5

# Load and preprocess data
data_path = "data/private_org_sops_20000.txt"
lines = tokenize_lines(data_path)
word2idx, idx2word = build_vocab(lines)
vocab_size = len(word2idx)

dataset = TextDataset(lines, word2idx, max_len=max_len + 1)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
global real_dataset
real_dataset = dataset  # For use in adversarial training

# Initialize models
generator = Generator(vocab_size, max_len=max_len).to(device)
discriminator = Discriminator(vocab_size, max_len=max_len).to(device)
rollout = Rollout(generator)

# Optimizers and loss
g_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
g_criterion = nn.NLLLoss()
d_criterion = nn.BCELoss()

# Pretrain Generator
pretrain_generator(generator, dataloader, nn.CrossEntropyLoss(), g_optimizer, num_epochs, device)

# Adversarial Training
for epoch in range(3):
    g_loss, d_loss = adversarial_train(generator, discriminator, rollout,
                                       g_optimizer, d_optimizer,
                                       g_criterion, d_criterion, device)
    print(f"[Adversarial] Epoch {epoch + 1}, G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")
