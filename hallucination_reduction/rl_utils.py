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
    HARD_PENALTY_IF_FACT_LT, MC_ROLLOUTS, GEN_LR, GEN_BATCH_SIZE, RL_EPOCHS, TOP_K, MAX_GEN_TOKENS, MIN_GEN_TOKENS, SAVE_DIR
)

# Device fallback: keep using env provided DEVICE if elsewhere; here choose cuda if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Helpers / protections
# -------------------------
def is_repetitive(text: str, repeat_threshold: float = 0.6):
    """
    Heuristic: if a single token occupies > repeat_threshold of tokens, treat as repetitive.
    Example: "Show Show Show Show" => repetitive.
    """
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
# unchanged compute_reward (only small protective tweak added)
# -------------------------
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

    # Penalize blatant repetition strongly
    if is_repetitive(generated_answer):
        combined = combined * 0.2  # heavy penalty for repeated outputs

    # clip
    combined = float(max(0.0, min(1.0, combined)))
    debug = {"p_fact": p_fact, "p_style": p_style, "p_safe": p_safe, "overlap": overlap_score, "combined": combined}
    return combined, debug

# -------------------------
# Monte-Carlo sampling (unchanged interface)
# -------------------------
def monte_carlo_rewards(prompt: str, generator, tokenizer, retrieved_docs, supporting_passages,
                        fact_disc, style_disc, safety_disc,
                        fact_tok, style_tok, safety_tok,
                        n_rollouts=MC_ROLLOUTS, device=DEVICE):
    # sample n_rollouts completions, return avg reward and list of (sample, reward, debug)
    # generate_answer expected to return list of generated strings when num_return_sequences > 1
    samples = generate_answer(generator, tokenizer, prompt,
                              max_new_tokens=MAX_GEN_TOKENS,
                              min_new_tokens=MIN_GEN_TOKENS,
                              num_return_sequences=n_rollouts,
                              device=device)

    rewards = []
    debug_list = []
    for s in samples:
        r, dbg = compute_reward(s, retrieved_docs, supporting_passages,
                                fact_disc, style_disc, safety_disc,
                                fact_tok, style_tok, safety_tok, device=device)
        rewards.append(r)
        debug_list.append({"sample": s, "debug": dbg})
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    return avg_reward, debug_list, samples

# -------------------------
# REINFORCE update with clipping & stability
# -------------------------
def reinforce_update(generator, tokenizer, prompts: List[str], sampled_texts: List[str], rewards: List[float],
                     gen_optimizer, baseline=0.0, device=DEVICE, max_adv=5.0, min_adv=-5.0, clip_grad_norm=1.0):
    """
    prompts, sampled_texts, rewards lists must be same length.
    Computes token log-prob of generated tokens only and runs REINFORCE with baseline.
    Added protections: advantage clipping, gradient clipping, try/except around backward.
    """
    if len(prompts) == 0:
        return 0.0

    generator.train()
    losses = []
    for prompt, sample, reward in zip(prompts, sampled_texts, rewards):
        full_text = prompt + sample
        # tokenization must be consistent with generation
        enc_full = tokenizer(full_text, return_tensors="pt", truncation=True).to(device)
        enc_prompt = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        prompt_len = enc_prompt.input_ids.shape[1]

        input_ids = enc_full.input_ids  # (1, L)
        try:
            outputs = generator(input_ids=input_ids, labels=input_ids)
        except RuntimeError as e:
            # if OOM or other exception happens, skip this sample to protect model
            print("Generator forward error in reinforce_update, skipping sample:", e)
            continue

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
        advantage = max(min(advantage, max_adv), min_adv)

        loss = -advantage * seq_log_prob
        losses.append(loss)

    if len(losses) == 0:
        return 0.0

    loss_tensor = torch.stack(losses).mean()

    gen_optimizer.zero_grad()
    try:
        loss_tensor.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_grad_norm)
        gen_optimizer.step()
    except Exception as e:
        print("Error during RL backward/step - skipping update to protect model:", e)
        gen_optimizer.zero_grad()
        return 0.0

    generator.eval()
    # return scalar loss
    try:
        return float(loss_tensor.detach().cpu().item())
    except Exception:
        return 0.0

# -------------------------
# Main RL loop (keeps your original signature)
# -------------------------
def reinforcement_learning_loop(generator, gen_tokenizer,
                                fact_disc, style_disc, safety_disc,
                                fact_tok, style_tok, safety_tok,
                                retriever, qa_pairs, device=DEVICE):
    # Set safe small LR for RL (override if original GEN_LR too big)
    safe_rl_lr = min(GEN_LR, 1e-6) if GEN_LR is not None else 1e-6
    gen_optimizer = optim.AdamW(generator.parameters(), lr=safe_rl_lr)

    # baseline moving average
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
            batch_prompts = []
            batch_samples = []
            batch_rewards = []
            batch_debugs = []

            for qa in batch:
                # build retrieval and prompt same as before
                retrieved_pairs = retriever.retrieve(qa.question, k=TOP_K)
                # retriever.retrieve returns list of (doc, sim) or maybe doc only in some impls; handle both:
                retrieved_docs = [p if isinstance(p, str) else p[0] for p in (retrieved_pairs if retrieved_pairs else [])]
                prompt = build_rag_prompt(qa.question, retrieved_docs)

                # run MC rollouts (protected)
                avg_reward, debug_list, samples = monte_carlo_rewards(
                    prompt, generator, gen_tokenizer, retrieved_docs, getattr(qa, "supporting_passages", retrieved_docs),
                    fact_disc, style_disc, safety_disc,
                    fact_tok, style_tok, safety_tok, n_rollouts=MC_ROLLOUTS, device=device
                )

                # if all samples are repetitive (bad), reduce avg_reward strongly
                repetitive_count = sum(1 for d in debug_list if is_repetitive(d["sample"]))
                if repetitive_count == len(debug_list) and len(debug_list) > 0:
                    avg_reward = avg_reward * 0.1  # penalize complete repetition

                # update moving baseline (use avg_reward)
                baseline = baseline_alpha * baseline + (1 - baseline_alpha) * avg_reward

                # append each sample from debug_list for update
                for d in debug_list:
                    batch_prompts.append(prompt)
                    batch_samples.append(d["sample"])
                    batch_rewards.append(d["debug"]["combined"])
                    batch_debugs.append(d["debug"])

            # If no samples collected, skip update
            if not batch_prompts:
                batch_index += 1
                continue

            # Normalize batch rewards (zero mean, unit std) to reduce variance
            rewards_arr = np.array(batch_rewards, dtype=float)
            if rewards_arr.size > 1:
                r_mean = rewards_arr.mean()
                r_std = rewards_arr.std() + 1e-8
                normalized_rewards = ((rewards_arr - r_mean) / r_std).tolist()
                # but we still keep original scale for reporting; use normalized rewards for update
                used_rewards = normalized_rewards
            else:
                used_rewards = rewards_arr.tolist()

            # policy update (use clipped advantages and gradient clipping inside)
            loss_val = reinforce_update(generator, gen_tokenizer, batch_prompts, batch_samples, used_rewards,
                                        gen_optimizer, baseline=baseline, device=device, max_adv=5.0, min_adv=-5.0, clip_grad_norm=1.0)

            history.append({"epoch": epoch, "batch_idx": batch_index, "loss": loss_val, "baseline": baseline, "mean_reward": float(np.mean(batch_rewards) if batch_rewards else 0.0)})
            epoch_rewards.extend(batch_rewards)

            if batch_index % 1 == 0:
                print(f"Epoch {epoch} batch {batch_index} loss {loss_val:.4f} baseline {baseline:.4f} mean_reward {(np.mean(batch_rewards) if batch_rewards else 0.0):.4f}")
                if batch_debugs:
                    # print only first debug sample to avoid huge logs
                    print("Sample debug:", batch_debugs[0])

            batch_index += 1

        # epoch end stats
        if epoch_rewards:
            epoch_mean = float(np.mean(epoch_rewards))
            epoch_std = float(np.std(epoch_rewards))
        else:
            epoch_mean = 0.0
            epoch_std = 0.0

        print(f"Epoch {epoch} reward stats: mean {epoch_mean:.4f}, std {epoch_std:.4f}")

        # checkpointing best
        ckpt_path = os.path.join(SAVE_DIR, f"generator_epoch_{epoch}.pt")
        try:
            torch.save(generator.state_dict(), ckpt_path)
            print("Saved generator checkpoint to", ckpt_path)
        except Exception as e:
            print("Warning: failed to save epoch checkpoint:", e)

        # Save best by mean reward
        if epoch_mean > best_mean_reward:
            best_mean_reward = epoch_mean
            best_ckpt = os.path.join(SAVE_DIR, f"generator_best_mean_{epoch}.pt")
            try:
                torch.save(generator.state_dict(), best_ckpt)
                print("Saved BEST checkpoint to", best_ckpt)
            except Exception as e:
                print("Warning: could not save best checkpoint:", e)

        # safety: if collapsed (very low mean) revert to best and stop
        if epoch_mean < 0.01:
            print("Mean reward collapsed extremely low — reverting to best checkpoint (if any) and stopping RL.")
            if best_ckpt and os.path.exists(best_ckpt):
                try:
                    generator.load_state_dict(torch.load(best_ckpt, map_location=device))
                    print("Restored best checkpoint:", best_ckpt)
                except Exception as e:
                    print("Failed to restore best checkpoint:", e)
            break

    # final save
    final_path = os.path.join(SAVE_DIR, "generator_final.pt")
    try:
        torch.save(generator.state_dict(), final_path)
        print("Saved final generator to", final_path)
    except Exception as e:
        print("Warning: failed to save final generator:", e)

    return history
