from .data_utils import build_dummy_corpus_and_qa
from .retriever import SimpleRetriever
from .generator import load_generator, sft_finetune_generator
from .discriminator import load_discriminator
from .discriminator_training_utils import train_discriminator_minibatch, evaluate_classifier
from .evaluation import evaluate_old_vs_new_generator
from .rl_utils import reinforcement_learning_loop
from .config import DEVICE, GEN_MODEL, DISC_MODEL, SAVE_DIR, SFT_EPOCHS, SFT_BATCH, SFT_LR, DISC_EPOCHS, DISC_BATCH, DISC_LR, RL_EPOCHS
import copy
import torch
import os
import random
import numpy as np
from torch import nn


# -------------------------
# Helper
# -------------------------
def unwrap(model):
    """Return the original model if wrapped in DataParallel."""
    return model.module if hasattr(model, "module") else model


# -------------------------
# Main runnable flow
# -------------------------
def main():
    # Build corpus & QA
    passages, qa_pairs = build_dummy_corpus_and_qa()
    retriever = SimpleRetriever(passages)

    # Load models
    print("Loading generator and discriminators...")
    gen_tok, generator = load_generator(GEN_MODEL, DEVICE)
    baseline_generator = copy.deepcopy(generator)

    fact_tok, fact_disc = load_discriminator(DISC_MODEL, DEVICE)
    style_tok, style_disc = load_discriminator(DISC_MODEL, DEVICE)
    safety_tok, safety_disc = load_discriminator(DISC_MODEL, DEVICE)

    # Enable gradient checkpointing on (large) generator to save memory
    try:
        if hasattr(generator, "gradient_checkpointing_enable"):
            generator.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing on generator.")
    except Exception as e:
        print("Could not enable gradient checkpointing:", e)

    # -------------------------
    # Multi-GPU support (generator only)
    # -------------------------
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 1:
        print(f"✅ Using {n_gpus} GPUs with DataParallel (generator only).")
        generator = nn.DataParallel(generator)
        # keep discriminators as single-device models (avoid small-model DataParallel overhead)
        fact_disc = fact_disc.to(DEVICE)
        style_disc = style_disc.to(DEVICE)
        safety_disc = safety_disc.to(DEVICE)
    else:
        print("Using single GPU or CPU.")
        generator = generator.to(DEVICE)
        fact_disc = fact_disc.to(DEVICE)
        style_disc = style_disc.to(DEVICE)
        safety_disc = safety_disc.to(DEVICE)

    # -------------------------
    # Build synthetic training data for discriminators
    # -------------------------
    fact_texts, fact_labels = [], []
    style_texts, style_labels = [], []
    safety_texts, safety_labels = [], []

    for qa in qa_pairs:
        # positive examples (gold answer)
        fact_texts.append(qa.answer); fact_labels.append(1)
        style_texts.append(qa.answer); style_labels.append(1)
        safety_texts.append(qa.answer); safety_labels.append(1)

        # fabricated negative
        fabricated = qa.answer + " Additionally, we permit indefinite third-party storage of sensitive logs."
        fact_texts.append(fabricated); fact_labels.append(0)
        safety_texts.append(fabricated); safety_labels.append(0)

        # style negative (rude/casual)
        bad_style = "nah, just do it. we don't care about tone."
        style_texts.append(bad_style); style_labels.append(0)

        # random negative noise (diversity)
        random_neg = "Misc unrelated content " + str(random.randint(0, 10000))
        fact_texts.append(random_neg); fact_labels.append(0)
        safety_texts.append(random_neg); safety_labels.append(0)
        style_texts.append(random_neg); style_labels.append(0)

    # -------------------------
    # Train discriminators
    # -------------------------
    print("Training fact discriminator...")
    fact_disc = train_discriminator_minibatch(fact_disc, fact_tok, fact_texts, fact_labels,
                                              device=DEVICE, epochs=DISC_EPOCHS, batch_size=DISC_BATCH, lr=DISC_LR)
    print("Training style discriminator...")
    style_disc = train_discriminator_minibatch(style_disc, style_tok, style_texts, style_labels,
                                               device=DEVICE, epochs=DISC_EPOCHS, batch_size=DISC_BATCH, lr=DISC_LR)
    print("Training safety discriminator...")
    safety_disc = train_discriminator_minibatch(safety_disc, safety_tok, safety_texts, safety_labels,
                                                device=DEVICE, epochs=DISC_EPOCHS, batch_size=DISC_BATCH, lr=DISC_LR)

    # -------------------------
    # Sanity check discriminator accuracy
    # -------------------------
    print("\nSanity-check discriminator metrics:")
    print("Fact disc:", evaluate_classifier(fact_disc, fact_tok, fact_texts, fact_labels, device=DEVICE))
    print("Style disc:", evaluate_classifier(style_disc, style_tok, style_texts, style_labels, device=DEVICE))
    print("Safety disc:", evaluate_classifier(safety_disc, safety_tok, safety_texts, safety_labels, device=DEVICE))

    # -------------------------
    # Evaluate baseline before any tuning
    # -------------------------
    print("\nEvaluation before SFT/RL (baseline):")
    _, old_before, _ = evaluate_old_vs_new_generator(baseline_generator, generator,
                                                     gen_tok, retriever, qa_pairs,
                                                     fact_disc, fact_tok, device=DEVICE)
    print("Baseline summary:", old_before)

    # -------------------------
    # Supervised Fine-Tuning (SFT)
    # -------------------------
    print("\nRunning supervised fine-tuning (SFT) on QA pairs...")
    generator = sft_finetune_generator(generator, gen_tok, qa_pairs,
                                       device=DEVICE, epochs=SFT_EPOCHS,
                                       batch_size=SFT_BATCH, lr=SFT_LR)

    print("\nEvaluation after SFT (before RL):")
    rows_sft, old_sft, new_sft = evaluate_old_vs_new_generator(baseline_generator, generator,
                                                               gen_tok, retriever, qa_pairs,
                                                               fact_disc, fact_tok, device=DEVICE)
    print("Old (baseline) summary:", old_sft)
    print("New (after SFT) summary:", new_sft)
    for r in rows_sft:
        print("\nQ:", r["question"])
        print("Gold:", r["gold"])
        print("Old:", r["old"])
        print("New:", r["new"])

    # -------------------------
    # Reinforcement Learning (RL)
    # -------------------------
    print("\nStarting Reinforcement Learning (REINFORCE)...")
    history = reinforcement_learning_loop(generator, gen_tok,
                                          fact_disc, style_disc, safety_disc,
                                          fact_tok, style_tok, safety_tok,
                                          retriever, qa_pairs, device=DEVICE)

    # -------------------------
    # Save final models
    # -------------------------
    torch.save(unwrap(generator).state_dict(), os.path.join(SAVE_DIR, "generator_final.pt"))
    torch.save(unwrap(fact_disc).state_dict(), os.path.join(SAVE_DIR, "fact_disc_final.pt"))
    torch.save(unwrap(style_disc).state_dict(), os.path.join(SAVE_DIR, "style_disc_final.pt"))
    torch.save(unwrap(safety_disc).state_dict(), os.path.join(SAVE_DIR, "safety_disc_final.pt"))
    print("✅ Models saved to", SAVE_DIR)

    # -------------------------
    # Final evaluation (old vs new)
    # -------------------------
    print("\nFinal Evaluation (baseline old vs new generator):")
    rows, old_summary, new_summary = evaluate_old_vs_new_generator(baseline_generator, generator,
                                                                   gen_tok, retriever, qa_pairs,
                                                                   fact_disc, fact_tok, device=DEVICE)
    print("\n=== Metrics Summary ===")
    print("Old (baseline):", old_summary)
    print("New (fine-tuned + RL):", new_summary)
    print("\n=== Side-by-side outputs ===")
    for r in rows:
        print("\nQ:", r["question"])
        print("Gold:", r["gold"])
        print("Old:", r["old"])
        print("New:", r["new"])
        print("-" * 60)

    print("\nHallucination rate reduction: {:.2%} -> {:.2%} (Δ {:.2%})".format(
        old_summary["hallucination_rate"],
        new_summary["hallucination_rate"],
        old_summary["hallucination_rate"] - new_summary["hallucination_rate"]
    ))


if __name__ == "__main__":
    main()
