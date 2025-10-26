import torch
from torch import optim, nn
import numpy as np
from typing import List
import random
import os
import math
from collections import Counter

from .discriminator import discriminator_predict_text
from .evaluation import overlap_fact_check
from .generator import build_rag_prompt, generate_answer
from .config import (
    FACT_WEIGHT, STYLE_WEIGHT, SAFETY_WEIGHT,
    HARD_PENALTY_IF_FACT_LT, MC_ROLLOUTS, GEN_LR, GEN_BATCH_SIZE,
    RL_EPOCHS, TOP_K, MAX_GEN_TOKENS, MIN_GEN_TOKENS, SAVE_DIR
)

# Device fallback: use cuda if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Helpers / protections
# -------------------------
def is_repetitive(text: str, repeat_threshold: float = 0.6):
    """Heuristic: if a single token occupies > repeat_threshold of tokens, treat as repetitive."""
    toks = text.split()
    if not toks:
        return False
    cnt = Counter(toks)
    most_common_frac = cnt.most_common(1)[0][1] / len(toks)
    return most_common_frac >= repeat_threshold


def safe_save_state(model, path):
    try:
        torch.save(model.state_dict(), path)
    except Exception as e:
        print("Warning: failed to save checkpoint:", e)


# -------------------------
# Compute reward
# -------------------------
def compute_reward(generated_answer: str, retrieved_docs: List[str], supporting_passages: List[str],
                   fact_disc, style_disc, safety_disc,
                   fact_tok, style_tok, safety_tok,
                   fact_weight=FACT_WEIGHT, style_weight=STYLE_WEIGHT, safety_weight=SAFETY_WEIGHT,
                   device=DEVICE):
    """Compute scalar reward combining factuality, style, safety, and overlap."""
    fact_res = discriminator_predict_text(fact_disc, fact_tok, [generated_answer], device=device)[0]
    style_res = discriminator_predict_text(style_disc, style_tok, [generated_answer], device=device)[0]
    safe_res = discriminator_predict_text(safety_disc, safety_tok, [generated_answer], device=device)[0]

    p_fact = fact_res["probs"][1] if len(fact_res["probs"]) > 1 else fact_res["probs"][0]
    p_style = style_res["probs"][1] if len(style_res["probs"]) > 1 else style_res["probs"][0]
    p_safe = safe_res["probs"][1] if len(safe_res["probs"]) > 1 else safe_res["probs"][0]

    overlap_score = overlap_fact_check(generated_answer, supporting_passages)

    # Weighted average (normalized)
    combined = fact_weight * p_fact + style_weight * p_style + safety_weight * p_safe
    total_w = fact_weight + style_weight + safety_weight
    combined /= total_w

    # Blend overlap and penalize low factuality
    combined = combined * 0.7 + overlap_score * 0.3
    if p_fact < 0.5:
        combined = max(0.0, combined - HARD_PENALTY_IF_FACT_LT)

    # Penalize repetition
    if is_repetitive(generated_answer):
        combined *= 0.2

    combined = float(max(0.0, min(1.0, combined)))
    debug = {"p_fact": p_fact, "p_style": p_style, "p_safe": p_safe,
             "overlap": overlap_score, "combined": combined}
    return combined, debug


# -------------------------
# Monte Carlo rollouts
# -------------------------
def monte_carlo_rewards(prompt: str, generator, tokenizer, retrieved_docs, supporting_passages,
                        fact_disc, style_disc, safety_disc,
                        fact_tok, style_tok, safety_tok,
                        n_rollouts=MC_ROLLOUTS, device=DEVICE):
    """Sample multiple generations and compute their average reward."""
    generator.eval()
    with torch.no_grad():
        samples = generate_answer(
            generator, tokenizer, prompt,
            max_new_tokens=MAX_GEN_TOKENS,
            min_new_tokens=MIN_GEN_TOKENS,
            num_return_sequences=n_rollouts,
            device=device
        )

    rewards, debug_list = [], []
    for s in samples:
        r, dbg = compute_reward(s, retrieved_docs, supporting_passages,
                                fact_disc, style_disc, safety_disc,
                                fact_tok, style_tok, safety_tok, device=device)
        rewards.append(r)
        debug_list.append({"sample": s, "debug": dbg})

    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    return avg_reward, debug_list, samples


# -------------------------
# Safer REINFORCE update
# -------------------------
def reinforce_update(generator, tokenizer, prompts: List[str], sampled_texts: List[str], rewards: List[float],
                     gen_optimizer, baseline=0.0, device=DEVICE,
                     max_adv=5.0, min_adv=-5.0, clip_grad_norm=1.0):
    """
    REINFORCE update with label masking and clipped advantages.
    Safer against tokenizer shifts and truncation.
    """
    if len(prompts) == 0:
        return 0.0

    generator.train()
    losses = []

    for prompt, sample, reward in zip(prompts, sampled_texts, rewards):
        # Ensure separation between prompt and sample
        sep = "\n\n"
        full_text = prompt + sep + sample

        enc_full = tokenizer(full_text, return_tensors="pt", truncation=True).to(device)
        enc_prompt = tokenizer(prompt + sep, return_tensors="pt", truncation=True).to(device)

        input_ids = enc_full.input_ids
        attention_mask = enc_full.attention_mask
        labels = input_ids.clone()
        prompt_len = enc_prompt.input_ids.shape[1]

        # Mask prompt tokens in labels
        labels[:, :prompt_len] = -100

        try:
            outputs = generator(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        except RuntimeError as e:
            print("Generator forward error, skipping sample:", e)
            continue

        logits = outputs.logits  # (1, L, V)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)

        loss_flat = nn.functional.cross_entropy(
            flat_logits, flat_labels, reduction="none", ignore_index=-100
        )
        loss_flat = loss_flat.view(shift_labels.size())
        mask = (shift_labels != -100).float()
        token_log_probs = -loss_flat * mask
        seq_log_prob = token_log_probs.sum()

        advantage = reward - baseline
        advantage = max(min(advantage, max_adv), min_adv)

        loss = -advantage * seq_log_prob
        losses.append(loss)

    if len(losses) == 0:
        return 0.0

    loss_tensor = torch.stack(losses).mean()

    gen_optimizer.zero_grad()
    try:
        loss_tensor.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_grad_norm)
        gen_optimizer.step()
    except Exception as e:
        print("Error during RL backward/step - skipping update:", e)
        gen_optimizer.zero_grad()
        return 0.0

    generator.eval()
    return float(loss_tensor.detach().cpu().item())


# -------------------------
# Main RL loop
# -------------------------
def reinforcement_learning_loop(generator, gen_tokenizer,
                                fact_disc, style_disc, safety_disc,
                                fact_tok, style_tok, safety_tok,
                                retriever, qa_pairs, device=DEVICE):
    """Main reinforcement learning loop for generator fine-tuning."""
    generator.to(device)
    fact_disc.to(device)
    style_disc.to(device)
    safety_disc.to(device)

    safe_rl_lr = min(GEN_LR, 1e-6) if GEN_LR is not None else 1e-6
    gen_optimizer = optim.AdamW(generator.parameters(), lr=safe_rl_lr)

    baseline = 0.5
    baseline_alpha = 0.9
    history = []
    best_mean_reward = -1.0
    best_ckpt = None

    for epoch in range(RL_EPOCHS):
        print(f"\n=== RL Epoch {epoch+1}/{RL_EPOCHS} ===")
        random.shuffle(qa_pairs)
        epoch_rewards = []
        batch_index = 0

        for i in range(0, len(qa_pairs), GEN_BATCH_SIZE):
            batch = qa_pairs[i:i+GEN_BATCH_SIZE]
            batch_prompts, batch_samples, batch_rewards, batch_debugs = [], [], [], []

            for qa in batch:
                retrieved_pairs = retriever.retrieve(qa.question, k=TOP_K)
                retrieved_docs = [p if isinstance(p, str) else p[0] for p in (retrieved_pairs or [])]
                prompt = build_rag_prompt(qa.question, retrieved_docs)

                avg_reward, debug_list, samples = monte_carlo_rewards(
                    prompt, generator, gen_tokenizer, retrieved_docs,
                    getattr(qa, "supporting_passages", retrieved_docs),
                    fact_disc, style_disc, safety_disc,
                    fact_tok, style_tok, safety_tok,
                    n_rollouts=MC_ROLLOUTS, device=device
                )

                repetitive_count = sum(1 for d in debug_list if is_repetitive(d["sample"]))
                if repetitive_count == len(debug_list) and len(debug_list) > 0:
                    avg_reward *= 0.1

                baseline = baseline_alpha * baseline + (1 - baseline_alpha) * avg_reward

                for d in debug_list:
                    batch_prompts.append(prompt)
                    batch_samples.append(d["sample"])
                    batch_rewards.append(d["debug"]["combined"])
                    batch_debugs.append(d["debug"])

            if not batch_prompts:
                batch_index += 1
                continue

            rewards_arr = np.array(batch_rewards, dtype=float)
            if rewards_arr.size > 1:
                r_mean, r_std = rewards_arr.mean(), rewards_arr.std() + 1e-8
                used_rewards = ((rewards_arr - r_mean) / r_std).tolist()
            else:
                used_rewards = rewards_arr.tolist()

            loss_val = reinforce_update(generator, gen_tokenizer,
                                        batch_prompts, batch_samples, used_rewards,
                                        gen_optimizer, baseline=baseline, device=device,
                                        max_adv=5.0, min_adv=-5.0, clip_grad_norm=1.0)

            history.append({
                "epoch": epoch, "batch_idx": batch_index,
                "loss": loss_val, "baseline": baseline,
                "mean_reward": float(np.mean(batch_rewards) if batch_rewards else 0.0)
            })
            epoch_rewards.extend(batch_rewards)

            print(f"Epoch {epoch} batch {batch_index} | loss {loss_val:.4f} | baseline {baseline:.4f} | "
                  f"mean_reward {(np.mean(batch_rewards) if batch_rewards else 0.0):.4f}")
            if batch_debugs:
                print("Sample debug:", batch_debugs[0])

            batch_index += 1

        epoch_mean = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        epoch_std = float(np.std(epoch_rewards)) if epoch_rewards else 0.0
        print(f"Epoch {epoch} reward stats: mean {epoch_mean:.4f}, std {epoch_std:.4f}")

        ckpt_path = os.path.join(SAVE_DIR, f"generator_epoch_{epoch}.pt")
        try:
            torch.save(generator.state_dict(), ckpt_path)
            print("Saved generator checkpoint to", ckpt_path)
        except Exception as e:
            print("Warning: failed to save epoch checkpoint:", e)

        if epoch_mean > best_mean_reward:
            best_mean_reward = epoch_mean
            best_ckpt = os.path.join(SAVE_DIR, f"generator_best_mean_{epoch}.pt")
            try:
                torch.save(generator.state_dict(), best_ckpt)
                print("Saved BEST checkpoint to", best_ckpt)
            except Exception as e:
                print("Warning: could not save best checkpoint:", e)

        if epoch_mean < 0.01:
            print("Mean reward collapsed — reverting to best checkpoint (if any).")
            if best_ckpt and os.path.exists(best_ckpt):
                try:
                    generator.load_state_dict(torch.load(best_ckpt, map_location=device))
                    print("Restored best checkpoint:", best_ckpt)
                except Exception as e:
                    print("Failed to restore best checkpoint:", e)
            break

    final_path = os.path.join(SAVE_DIR, "generator_final.pt")
    try:
        torch.save(generator.state_dict(), final_path)
        print("Saved final generator to", final_path)
    except Exception as e:
        print("Warning: failed to save final generator:", e)

    return history
