import torch
import torch.nn as nn
import torch.nn.functional as F

def pretrain_generator(generator, dataloader, criterion, optimizer, num_epochs, device):
    generator.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits, _ = generator(inputs)
            loss = criterion(logits.view(-1, generator.vocab_size), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Pretrain G] Epoch {epoch + 1}, Loss: {total_loss:.4f}")

def train_discriminator(discriminator, real_data, fake_data, optimizer, criterion, device):
    discriminator.train()
    optimizer.zero_grad()
    real_labels = torch.ones(real_data.size(0), 1).to(device)
    fake_labels = torch.zeros(fake_data.size(0), 1).to(device)
    real_preds = discriminator(real_data)
    fake_preds = discriminator(fake_data)
    loss = criterion(real_preds, real_labels) + criterion(fake_preds, fake_labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def adversarial_train(generator, discriminator, rollout, g_optimizer, d_optimizer, g_criterion, d_criterion, device):
    fake_samples = generator.sample(batch_size=64, device=device)
    rewards = rollout.get_reward(fake_samples, discriminator, rollout_num=16, device=device)

    # Generator loss
    logits, _ = generator(fake_samples)
    log_probs = F.log_softmax(logits, dim=-1)
    g_loss = 0
    for t in range(fake_samples.size(1)):
        token_log_prob = log_probs[:, t, :].gather(1, fake_samples[:, t].unsqueeze(1)).squeeze()
        g_loss += -torch.mean(token_log_prob * rewards[:, t])
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    # Discriminator training
    real_data = real_dataset[torch.randint(0, len(real_dataset), (64,))].to(device)
    d_loss = train_discriminator(discriminator, real_data, fake_samples.detach(), d_optimizer, d_criterion, device)

    rollout.update_params()
    return g_loss.item(), d_loss
