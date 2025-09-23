# hallucination_reduction/main.py

import os
import torch
import copy

from .data_utils import load_dataset
from .retriever import SimpleRetriever
from .generator import load_generator, generate_answer, train_generator_minibatch
from .discriminator import load_discriminator
from .discriminator_training_utils import train_all_discriminators
from .evaluation import evaluate_old_vs_new_generator
from .rl_utils import reinforce_train_step
from .config import (
    DEVICE, SFT_EPOCHS, SFT_BATCH, SFT_LR,
    DISC_EPOCHS, DISC_BATCH, DISC_LR,
    RL_EPOCHS, GEN_LR, SAVE_DIR
)


def main():
    # -------- 1. Load dataset --------
    dataset_path = "data/raw/real_qa.json"
    passages, qa_pairs = load_dataset(dataset_path)
    retriever = SimpleRetriever(passages)

    # -------- 2. Load generator --------
    gen_tok, generator = load_generator()
    generator = generator.to(DEVICE)
    baseline_generator = copy.deepcopy(generator).to(DEVICE)

    # -------- 3. Load discriminators --------
    fact_tok, fact_disc = load_discriminator()
    style_tok, style_disc = load_discriminator()
    safety_tok, safety_disc = load_discriminator()
    fact_disc = fact_disc.to(DEVICE)
    style_disc = style_disc.to(DEVICE)
    safety_disc = safety_disc.to(DEVICE)

    # -------- 4. Train or load discriminators --------
    fact_path = os.path.join(SAVE_DIR, "fact_disc")
    style_path = os.path.join(SAVE_DIR, "style_disc")
    safety_path = os.path.join(SAVE_DIR, "safety_disc")

    if all(os.path.exists(p) for p in [fact_path, style_path, safety_path]):
        print("Loading trained discriminators...")
        fact_tok = type(fact_tok).from_pretrained(fact_path)
        fact_disc = type(fact_disc).from_pretrained(fact_path).to(DEVICE)
        style_tok = type(style_tok).from_pretrained(style_path)
        style_disc = type(style_disc).from_pretrained(style_path).to(DEVICE)
        safety_tok = type(safety_tok).from_pretrained(safety_path)
        safety_disc = type(safety_disc).from_pretrained(safety_path).to(DEVICE)
    else:
        print("Training discriminators...")
        fact_disc, style_disc, safety_disc = train_all_discriminators(
            fact_disc, fact_tok,
            style_disc, style_tok,
            safety_disc, safety_tok,
            qa_pairs,
            device=DEVICE, epochs=DISC_EPOCHS, batch_size=DISC_BATCH, lr=DISC_LR
        )
        fact_tok.save_pretrained(fact_path)
        fact_disc.save_pretrained(fact_path)
        style_tok.save_pretrained(style_path)
        style_disc.save_pretrained(style_path)
        safety_tok.save_pretrained(safety_path)
        safety_disc.save_pretrained(safety_path)

    # -------- 5. Baseline evaluation --------
    print("\nEvaluation before SFT/RL:")
    rows, old_summary, new_summary = evaluate_old_vs_new_generator(
        baseline_generator, generator, gen_tok,
        retriever, qa_pairs, fact_disc, fact_tok,
        device=DEVICE
    )
    print("Baseline summary:", old_summary)

    # -------- 6. Supervised Fine-Tuning (SFT) --------
    gen_sft_path = os.path.join(SAVE_DIR, "generator_sft")
    if os.path.exists(gen_sft_path):
        print("Loading SFT fine-tuned generator...")
        gen_tok = type(gen_tok).from_pretrained(gen_sft_path)
        generator = type(generator).from_pretrained(gen_sft_path).to(DEVICE)
    else:
        print("\nRunning Supervised Fine-Tuning (SFT)...")
        generator = train_generator_minibatch(
            generator, gen_tok, qa_pairs,
            epochs=SFT_EPOCHS, batch_size=SFT_BATCH, lr=SFT_LR, device=DEVICE
        )
        gen_tok.save_pretrained(gen_sft_path)
        generator.save_pretrained(gen_sft_path)

    # -------- 7. RL Fine-Tuning --------
    gen_rl_path = os.path.join(SAVE_DIR, "generator_rl")
    if os.path.exists(gen_rl_path):
        print("Loading RL fine-tuned generator...")
        gen_tok = type(gen_tok).from_pretrained(gen_rl_path)
        generator = type(generator).from_pretrained(gen_rl_path).to(DEVICE)
    else:
        print("\nRunning Reinforcement Learning (RL)...")
        optimizer = torch.optim.Adam(generator.parameters(), lr=GEN_LR)

        for epoch in range(RL_EPOCHS):
            total_loss, total_reward = 0.0, 0.0
            for qa in qa_pairs:
                loss, reward, gen_answer = reinforce_train_step(
                    generator, gen_tok, qa,
                    fact_disc, style_disc, safety_disc,
                    fact_tok, style_tok, safety_tok,
                    optimizer, device=DEVICE
                )
                total_loss += loss
                total_reward += reward

            avg_loss = total_loss / len(qa_pairs)
            avg_reward = total_reward / len(qa_pairs)
            print(f"RL Epoch {epoch+1}/{RL_EPOCHS} - Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")

        gen_tok.save_pretrained(gen_rl_path)
        generator.save_pretrained(gen_rl_path)

    # -------- 8. Final evaluation --------
    print("\nFinal Evaluation after SFT + RL:")
    rows, old_summary, new_summary = evaluate_old_vs_new_generator(
        baseline_generator, generator, gen_tok,
        retriever, qa_pairs, fact_disc, fact_tok,
        device=DEVICE
    )
    print("Final summary:", new_summary)


if __name__ == "__main__":
    main()
