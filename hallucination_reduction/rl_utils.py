# hallucination_reduction/rl_utils.py
import torch
import torch.nn.functional as F
import numpy as np
from .discriminator import discriminator_predict_text

HARD_PENALTY_IF_FACT_LT = 0.4
FACT_WEIGHT = 0.8
STYLE_WEIGHT = 0.15
SAFETY_WEIGHT = 0.05

def overlap_fact_check(answer, supporting_passages):
    answer_words = set(answer.lower().split())
    passage_words = set(" ".join(supporting_passages).lower().split())
    if not answer_words:
        return 0.0
    return len(answer_words & passage_words) / len(answer_words)

def compute_reward(generated_answer, supporting_passages,
                   fact_disc, style_disc, safety_disc,
                   fact_tok, style_tok, safety_tok,
                   fact_weight=FACT_WEIGHT, style_weight=STYLE_WEIGHT,
                   safety_weight=SAFETY_WEIGHT, device="cpu"):
    """
    Return shaped reward in [0,1] but clipped to a narrow band to avoid RL collapse.
    """
    fact_res = discriminator_predict_text(fact_disc, fact_tok, [generated_answer], device=device)[0]
    style_res = discriminator_predict_text(style_disc, style_tok, [generated_answer], device=device)[0]
    safe_res = discriminator_predict_text(safety_disc, safety_tok, [generated_answer], device=device)[0]

    p_fact = fact_res["probs"][1] if len(fact_res["probs"]) > 1 else fact_res["probs"][0]
    p_style = style_res["probs"][1] if len(style_res["probs"]) > 1 else style_res["probs"][0]
    p_safe = safe_res["probs"][1] if len(safe_res["probs"]) > 1 else safe_res["probs"][0]

    overlap_score = overlap_fact_check(generated_answer, supporting_passages)

    combined = (fact_weight * p_fact + style_weight * p_style + safety_weight * p_safe)
    combined = combined / (fact_weight + style_weight + safety_weight)

    # mix with overlap signal
    combined = 0.7 * combined + 0.3 * overlap_score

    # clip into a narrow band to stabilize gradients (adjustable)
    combined = float(max(0.4, min(0.6, combined)))

    # penalize strongly if fact prob low
    if p_fact < 0.5:
        combined = max(0.0, combined - HARD_PENALTY_IF_FACT_LT)

    return combined

def reinforce_train_step(generator, tokenizer, qa_pair,
                         fact_disc, style_disc, safety_disc,
                         fact_tok, style_tok, safety_tok,
                         optimizer, device="cpu", baseline=0.5, clip_adv=(-5.0, 5.0)):
    """
    Single REINFORCE update for one QA pair.
    - Uses token-level log-probs from the model to compute sequence log-prob for generated tokens only.
    - Applies advantage = reward - baseline.
    """
    generator.train()

    # Build prompt and move to device
    prompt = f"Question: {qa_pair.question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    # Sample one generation (deterministic sampling settings can be changed)
    gen_ids = generator.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        top_k=50,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )[0].to(device)  # shape (L_full,)

    # Decode generated answer text for reward
    gen_answer = tokenizer.decode(gen_ids[inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    # Compute reward (float in [0,1])
    reward = compute_reward(gen_answer, qa_pair.supporting_passages,
                            fact_disc, style_disc, safety_disc,
                            fact_tok, style_tok, safety_tok,
                            device=device)

    # Compute token-wise log-probs for full generated sequence
    full_input_ids = gen_ids.unsqueeze(0)  # (1, L)
    outputs = generator(input_ids=full_input_ids)
    logits = outputs.logits  # (1, L, V)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = full_input_ids[..., 1:].contiguous()

    # token-wise cross-entropy (no reduction)
    vocab_size = shift_logits.size(-1)
    loss_flat = F.cross_entropy(shift_logits.view(-1, vocab_size),
                                shift_labels.view(-1),
                                reduction='none')  # ( (L-1) * 1, )
    log_probs = -loss_flat  # per-token log-probs
    log_probs = log_probs.view(shift_labels.size())  # (1, L-1)

    # identify index where generated tokens start (for shift_labels)
    prompt_len = inputs['input_ids'].shape[1]
    gen_start_idx = max(0, prompt_len - 1)  # as shift_labels align with input_ids shifted by 1

    gen_token_log_probs = log_probs[0, gen_start_idx:]
    seq_log_prob = gen_token_log_probs.sum()  # scalar

    # advantage = reward - baseline (centered)
    advantage = reward - baseline
    advantage = max(min(advantage, clip_adv[1]), clip_adv[0])

    # REINFORCE objective (minimize -adv * log_prob)
    loss = -advantage * seq_log_prob

    optimizer.zero_grad()
    # loss is a torch scalar (seq_log_prob is torch tensor), but ensure grad path
    loss.backward()
    optimizer.step()

    generator.eval()
    return loss.item(), reward, gen_answer
