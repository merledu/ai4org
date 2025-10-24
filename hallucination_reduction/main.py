# hallucination_reduction/main.py
from .data_utils import build_dummy_corpus_and_qa
from .retriever import SimpleRetriever
from .generator import load_generator, sft_finetune_generator
from .discriminator import load_discriminator
from .discriminator_training_utils import train_discriminator_minibatch, evaluate_classifier
from .evaluation import evaluate_old_vs_new_generator
from .rl_utils import reinforcement_learning_loop
from .config import (
    GEN_MODEL,
    DISC_MODEL,
    SAVE_DIR,
    SFT_EPOCHS,
    SFT_BATCH,
    SFT_LR,
    DISC_EPOCHS,
    DISC_BATCH,
    DISC_LR,
    RL_EPOCHS,
)
import copy
import torch
import os
import random


# -------------------------
# Helper
# -------------------------
def unwrap(model):
    """Return the original model if wrapped in DataParallel or similar."""
    return model.module if hasattr(model, "module") else model


# -------------------------
# Main runnable flow
# -------------------------
def main():
    # -------------------------
    # Detect device (for discriminators / small models)
    # -------------------------
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    print(f"🔥 Using device for discriminators / CPU fallback: {DEVICE}")

    # -------------------------
    # Build corpus & QA
    # -------------------------
    passages, qa_pairs = build_dummy_corpus_and_qa()
    retriever = SimpleRetriever(passages)

    # -------------------------
    # Load models
    # -------------------------
    print("Loading generator and discriminators...")
    # Prefer to load generator with device_map='auto' if load_generator supports it.
    try:
        gen_tok, generator = load_generator(GEN_MODEL, device_map="auto")
        print("Loaded generator with device_map='auto' (accelerate will shard across GPUs).")
    except TypeError:
        # fallback to older signature
        gen_tok, generator = load_generator(GEN_MODEL, DEVICE)
        print("Loaded generator (fallback call).")

    baseline_generator = copy.deepcopy(generator)

    # discriminators (small models) -> move to single device
    fact_tok, fact_disc = load_discriminator(DISC_MODEL, DEVICE)
    style_tok, style_disc = load_discriminator(DISC_MODEL, DEVICE)
    safety_tok, safety_disc = load_discriminator(DISC_MODEL, DEVICE)

    fact_disc = fact_disc.to(DEVICE)
    style_disc = style_disc.to(DEVICE)
    safety_disc = safety_disc.to(DEVICE)

    # Enable gradient checkpointing to save VRAM (if supported)
    if hasattr(generator, "gradient_checkpointing_enable"):
        try:
            generator.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing on generator.")
        except Exception as e:
            print("Could not enable gradient checkpointing:", e)

    # Info about multi-GPU
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 1:
        print(f"✅ Detected {n_gpus} GPUs. If generator used device_map='auto', it will be sharded across GPUs.")
    else:
        print("Using single GPU or CPU for discriminators and fallback.")

    # -------------------------
    # Build synthetic training data for discriminators
    # -------------------------
    fact_texts, fact_labels = [], []
    style_texts, style_labels = [], []
    safety_texts, safety_labels = [], []

    for qa in qa_pairs:
        # Positive examples
        fact_texts.append(qa.answer)
        fact_labels.append(1)
        style_texts.append(qa.answer)
        style_labels.append(1)
        safety_texts.append(qa.answer)
        safety_labels.append(1)

        # Fabricated / negative examples
        fabricated = qa.answer + " Additionally, we permit indefinite third-party storage of sensitive logs."
        fact_texts.append(fabricated)
        fact_labels.append(0)
        safety_texts.append(fabricated)
        safety_labels.append(0)

        bad_style = "nah, just do it. we don't care about tone."
        style_texts.append(bad_style)
        style_labels.append(0)

        random_neg = "Misc unrelated content " + str(random.randint(0, 10000))
        fact_texts.append(random_neg)
        fact_labels.append(0)
        safety_texts.append(random_neg)
        safety_labels.append(0)
        style_texts.append(random_neg)
        style_labels.append(0)

    # -------------------------
    # Train discriminators
    # -------------------------
    print("Training fact discriminator...")
    fact_disc = train_discriminator_minibatch(
        fact_disc, fact_tok, fact_texts, fact_labels, device=DEVICE, epochs=DISC_EPOCHS, batch_size=DISC_BATCH, lr=DISC_LR
    )

    print("Training style discriminator...")
    style_disc = train_discriminator_minibatch(
        style_disc, style_tok, style_texts, style_labels, device=DEVICE, epochs=DISC_EPOCHS, batch_size=DISC_BATCH, lr=DISC_LR
    )

    print("Training safety discriminator...")
    safety_disc = train_discriminator_minibatch(
        safety_disc, safety_tok, safety_texts, safety_labels, device=DEVICE, epochs=DISC_EPOCHS, batch_size=DISC_BATCH, lr=DISC_LR
    )

    # -------------------------
    # Sanity check discriminator accuracy
    # -------------------------
    print("\nSanity-check discriminator metrics:")
    print("Fact disc:", evaluate_classifier(fact_disc, fact_tok, fact_texts, fact_labels, device=DEVICE))
    print("Style disc:", evaluate_classifier(style_disc, style_tok, style_texts, style_labels, device=DEVICE))
    print("Safety disc:", evaluate_classifier(safety_disc, safety_tok, safety_texts, safety_labels, device=DEVICE))

    # -------------------------
    # Evaluation before tuning (baseline)
    # -------------------------
    print("\nEvaluation before SFT/RL (baseline):")
    # Important: do NOT move the generator if it's accelerate-dispatched; pass as-is.
    _, old_before, _ = evaluate_old_vs_new_generator(
        baseline_generator, generator, gen_tok, retriever, qa_pairs, fact_disc, fact_tok, device=DEVICE
    )
    print("Baseline summary:", old_before)

    # -------------------------
    # Supervised Fine-Tuning (SFT)
    # -------------------------
    print("\nRunning supervised fine-tuning (SFT) on QA pairs...")
    generator = sft_finetune_generator(generator, gen_tok, qa_pairs, device=DEVICE, epochs=SFT_EPOCHS, batch_size=SFT_BATCH, lr=SFT_LR)

    # -------------------------
    # Evaluation after SFT
    # -------------------------
    print("\nEvaluation after SFT (before RL):")
    _, old_sft, new_sft = evaluate_old_vs_new_generator(
        baseline_generator, generator, gen_tok, retriever, qa_pairs, fact_disc, fact_tok, device=DEVICE
    )
    print("Old (baseline) summary:", old_sft)
    print("New (after SFT) summary:", new_sft)

    # -------------------------
    # Reinforcement Learning (RL)
    # -------------------------
    print("\nStarting Reinforcement Learning (REINFORCE)...")
    history = reinforcement_learning_loop(
        generator, gen_tok, fact_disc, style_disc, safety_disc, fact_tok, style_tok, safety_tok, retriever, qa_pairs, device=DEVICE
    )

    # -------------------------
    # Save final models
    # -------------------------
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(unwrap(generator).state_dict(), os.path.join(SAVE_DIR, "generator_final.pt"))
    torch.save(unwrap(fact_disc).state_dict(), os.path.join(SAVE_DIR, "fact_disc_final.pt"))
    torch.save(unwrap(style_disc).state_dict(), os.path.join(SAVE_DIR, "style_disc_final.pt"))
    torch.save(unwrap(safety_disc).state_dict(), os.path.join(SAVE_DIR, "safety_disc_final.pt"))
    print("✅ Models saved to", SAVE_DIR)

    # -------------------------
    # Final evaluation
    # -------------------------
    print("\nFinal Evaluation (baseline old vs new generator):")
    rows, old_summary, new_summary = evaluate_old_vs_new_generator(
        baseline_generator, generator, gen_tok, retriever, qa_pairs, fact_disc, fact_tok, device=DEVICE
    )

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

    print(
        "\nHallucination rate reduction: {:.2%} -> {:.2%} (Δ {:.2%})".format(
            old_summary["hallucination_rate"],
            new_summary["hallucination_rate"],
            old_summary["hallucination_rate"] - new_summary["hallucination_rate"],
        )
    )


if __name__ == "__main__":
    main()
