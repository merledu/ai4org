import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class Rollout:
    def __init__(self, generator, update_rate=0.8):
        self.generator = generator
        self.udpate_rate = update_rate
        self.rolling_generator = deepcopy(generator)

    def get_reward(self, samples, discriminator, rollout_num, device):
        rewards = []
        batch_size, seq_len = samples.size()

        for t in range(1, seq_len + 1):
            temp_samples = []
            for _ in range(rollout_num):
                partial = samples[:, :t]
                rest = self._rollout(partial, seq_len - t, device)
                full = torch.cat([partial, rest], dim=1)
                temp_samples.append(full)

            all_samples = torch.cat(temp_samples, dim=0)
            scores = discriminator(all_samples).detach()
            rewards_t = scores.view(rollout_num, batch_size).mean(dim=0)
            rewards.append(rewards_t)

        return torch.stack(rewards, dim=1)

    def _rollout(self, partial_seq, remaining_len, device):
        self.rolling_generator.eval()
        samples = partial_seq
        h, c = None, None
        for _ in range(remaining_len):
            logits, (h, c) = self.rolling_generator(samples, (h, c) if h is not None else None)
            next_token = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), 1)
            samples = torch.cat([samples, next_token], dim=1)
        return samples[:, partial_seq.size(1):]

    def update_params(self):
        for target_param, source_param in zip(self.rolling_generator.parameters(), self.generator.parameters()):
            target_param.data = self.udpate_rate * target_param.data + (1 - self.udpate_rate) * source_param.data
