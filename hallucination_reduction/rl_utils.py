import torch
from torch import optim,nn
import numpy as np
from typing import List
import random
import os

from .discriminator import discriminator_predict_text
from .evaluation import overlap_fact_check
from .generator import build_rag_prompt, generate_answer
from .config import (
     FACT_WEIGHT, STYLE_WEIGHT, SAFETY_WEIGHT,
    HARD_PENALTY_IF_FACT_LT, MC_ROLLOUTS, GEN_LR, GEN_BATCH_SIZE, RL_EPOCHS, TOP_K, MAX_GEN_TOKENS, MIN_GEN_TOKENS, SAVE_DIR
)

DEVICE="cuda"

def compute_reward(generated_answer: str, retrieved_docs: List[str], supporting_passages: List[str],
                   fact_disc, style_disc, safety_disc,
                   fact_tok, style_tok, safety_tok,
                   fact_weight=FACT_WEIGHT, style_weight=STYLE_WEIGHT, safety_weight=SAFETY_WEIGHT,
                   device=DEVICE):
    # probabilities of positive class (index 1)
    fact_res = discriminator_predict_text(fact_disc, fact_tok, [generated_answer], device=device)[0]
    style_res = discriminator_predict_text(style_disc, style_tok, [generated_answer], device=device)[0]
    safe_res = discriminator_predict_text(safety_disc, safety_tok, [generated_answer], device=device)[0]

    p_fact = fact_res["probs"][1] if len(fact_res["probs"])>1 else fact_res["probs"][0]
    p_style = style_res["probs"][1] if len(style_res["probs"])>1 else style_res["probs"][0]
    p_safe = safe_res["probs"][1] if len(safe_res["probs"])>1 else safe_res["probs"][0]

    # overlap check
    overlap_score = overlap_fact_check(generated_answer, supporting_passages)

    # Weighted average (in [0,1])
    combined = fact_weight * p_fact + style_weight * p_style + safety_weight * p_safe
    total_w = fact_weight + style_weight + safety_weight
    combined = combined / total_w

    # Add small bonus for overlap and penalize low fact_prob strongly
    combined = combined * 0.7 + overlap_score * 0.3
    if p_fact < 0.5:
        combined = max(0.0, combined - HARD_PENALTY_IF_FACT_LT)

    # clip
    combined = float(max(0.0, min(1.0, combined)))
    debug = {"p_fact": p_fact, "p_style": p_style, "p_safe": p_safe, "overlap": overlap_score, "combined": combined}
    return combined, debug

def monte_carlo_rewards(prompt: str, generator, tokenizer, retrieved_docs, supporting_passages,
                        fact_disc, style_disc, safety_disc,
                        fact_tok, style_tok, safety_tok,
                        n_rollouts=MC_ROLLOUTS, device=DEVICE):
    # sample n_rollouts completions, return avg reward and list of (sample, reward, debug)
    samples = generate_answer(generator, tokenizer, prompt, max_new_tokens=MAX_GEN_TOKENS, min_new_tokens=MIN_GEN_TOKENS, num_return_sequences=n_rollouts, device=device)
    rewards = []
    debug_list = []
    for s in samples:
        r, dbg = compute_reward(s, retrieved_docs, supporting_passages, fact_disc, style_disc, safety_disc, fact_tok, style_tok, safety_tok, device=device)
        rewards.append(r)
        debug_list.append({"sample": s, "debug": dbg})
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    return avg_reward, debug_list, samples

def reinforce_update(generator, tokenizer, prompts: List[str], sampled_texts: List[str], rewards: List[float],
                     gen_optimizer, baseline=0.0, device=DEVICE):
    """
    prompts, sampled_texts, rewards lists must be same length.
    Computes token log-prob of generated tokens only and runs REINFORCE with baseline.
    """
    if len(prompts) == 0:
        return 0.0
    generator.train()
    losses = []
    for prompt, sample, reward in zip(prompts, sampled_texts, rewards):
        full_text = prompt + sample
        enc_full = tokenizer(full_text, return_tensors="pt", truncation=True).to(device)
        enc_prompt = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        prompt_len = enc_prompt.input_ids.shape[1]
        input_ids = enc_full.input_ids  # (1, L)
        outputs = generator(input_ids=input_ids, labels=input_ids)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        # token-wise cross entropy losses -> negative is log-prob
        loss_flat = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                                shift_labels.view(-1), reduction="none")
        log_probs = -loss_flat
        log_probs = log_probs.view(shift_labels.size())  # (1, L-1)
        gen_start_idx = max(0, prompt_len - 1)
        gen_token_log_probs = log_probs[0, gen_start_idx:]
        seq_log_prob = gen_token_log_probs.sum()
        advantage = reward - baseline
        # clip advantage for stability
        advantage = max(min(advantage, 5.0), -5.0)
        loss = -advantage * seq_log_prob
        losses.append(loss)
    loss_tensor = torch.stack(losses).mean()
    gen_optimizer.zero_grad()
    loss_tensor.backward()
    gen_optimizer.step()
    generator.eval()
    return loss_tensor.item()


def reinforcement_learning_loop(generator, gen_tokenizer,
                                fact_disc, style_disc, safety_disc,
                                fact_tok, style_tok, safety_tok,
                                retriever, qa_pairs, device=DEVICE):
    gen_optimizer = optim.AdamW(generator.parameters(), lr=GEN_LR)
    baseline = 0.5  # initial baseline in [0,1]
    baseline_alpha = 0.9
    history = []
    for epoch in range(RL_EPOCHS):
        print(f"\n=== RL Epoch {epoch+1}/{RL_EPOCHS} ===")
        random.shuffle(qa_pairs)
        epoch_rewards = []
        batch_index = 0
        for i in range(0, len(qa_pairs), GEN_BATCH_SIZE):
            batch = qa_pairs[i:i+GEN_BATCH_SIZE]
            batch_prompts = []
            batch_samples = []
            batch_rewards = []
            batch_debugs = []

            for qa in batch:
                retrieved = [p for idx,p in retriever.retrieve(qa.question, k=TOP_K)]
                prompt = build_rag_prompt(qa.question, retrieved)
                avg_reward, debug_list, samples = monte_carlo_rewards(
                    prompt, generator, gen_tokenizer, retrieved, qa.supporting_passages,
                    fact_disc, style_disc, safety_disc,
                    fact_tok, style_tok, safety_tok, n_rollouts=MC_ROLLOUTS, device=device
                )
                # update baseline (moving average)
                baseline = baseline_alpha * baseline + (1 - baseline_alpha) * avg_reward
                for d in debug_list:
                    batch_prompts.append(prompt)
                    batch_samples.append(d["sample"])
                    batch_rewards.append(d["debug"]["combined"])
                    batch_debugs.append(d["debug"])
            # policy update
            loss_val = reinforce_update(generator, gen_tokenizer, batch_prompts, batch_samples, batch_rewards, gen_optimizer, baseline=baseline, device=device)
            history.append({"epoch": epoch, "batch_idx": batch_index, "loss": loss_val, "baseline": baseline, "mean_reward": float(np.mean(batch_rewards) if batch_rewards else 0.0)})
            epoch_rewards.extend(batch_rewards)
            if batch_index % 1 == 0:
                print(f"Epoch {epoch} batch {batch_index} loss {loss_val:.4f} baseline {baseline:.4f} mean_reward {(np.mean(batch_rewards) if batch_rewards else 0.0):.4f}")
                if batch_debugs:
                    print("Sample debug:", batch_debugs[0])
            batch_index += 1
        print(f"Epoch {epoch} reward stats: mean {np.mean(epoch_rewards):.4f}, std {np.std(epoch_rewards):.4f}")
        # checkpoint
        ckpt = os.path.join(SAVE_DIR, f"generator_epoch_{epoch}.pt")
        torch.save(generator.state_dict(), ckpt)
        print("Saved generator checkpoint to", ckpt)
    torch.save(generator.state_dict(), os.path.join(SAVE_DIR, "generator_final.pt"))
    return history
