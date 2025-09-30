# main.py
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
from torch import nn, optim


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

    # Build synthetic training data for discriminators (bigger)
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
        random_neg = "Misc unrelated content " + str(random.randint(0,10000))
        fact_texts.append(random_neg); fact_labels.append(0)
        safety_texts.append(random_neg); safety_labels.append(0)
        style_texts.append(random_neg); style_labels.append(0)

    # Train discriminators with validation
    print("Training fact discriminator...")
    fact_disc = train_discriminator_minibatch(fact_disc, fact_tok, fact_texts, fact_labels, device=DEVICE, epochs=DISC_EPOCHS, batch_size=DISC_BATCH, lr=DISC_LR)
    print("Training style discriminator...")
    style_disc = train_discriminator_minibatch(style_disc, style_tok, style_texts, style_labels, device=DEVICE, epochs=DISC_EPOCHS, batch_size=DISC_BATCH, lr=DISC_LR)
    print("Training safety discriminator...")
    safety_disc = train_discriminator_minibatch(safety_disc, safety_tok, safety_texts, safety_labels, device=DEVICE, epochs=DISC_EPOCHS, batch_size=DISC_BATCH, lr=DISC_LR)

    # Sanity-check discriminator accuracy on whole synthetic set
    print("\nSanity-check discriminator metrics:")
    print("Fact disc:", evaluate_classifier(fact_disc, fact_tok, fact_texts, fact_labels, device=DEVICE))
    print("Style disc:", evaluate_classifier(style_disc, style_tok, style_texts, style_labels, device=DEVICE))
    print("Safety disc:", evaluate_classifier(safety_disc, safety_tok, safety_texts, safety_labels, device=DEVICE))

    # Evaluate baseline before any tuning
    print("\nEvaluation before SFT/RL (baseline):")
    _, old_before, _ = evaluate_old_vs_new_generator(baseline_generator, generator, gen_tok, retriever, qa_pairs, fact_disc, fact_tok, device=DEVICE)
    print("Baseline summary:", old_before)

    # Supervised fine-tuning (SFT)
    print("\nRunning supervised fine-tuning (SFT) on QA pairs...")
    generator = sft_finetune_generator(generator, gen_tok, qa_pairs, device=DEVICE, epochs=SFT_EPOCHS, batch_size=SFT_BATCH, lr=SFT_LR)

    print("\nEvaluation after SFT (before RL):")
    rows_sft, old_sft, new_sft = evaluate_old_vs_new_generator(baseline_generator, generator, gen_tok, retriever, qa_pairs, fact_disc, fact_tok, device=DEVICE)
    print("Old (baseline) summary:", old_sft)
    print("New (after SFT) summary:", new_sft)
    for r in rows_sft:
        print("\nQ:", r["question"])
        print("Gold:", r["gold"])
        print("Old:", r["old"])
        print("New:", r["new"])

    # RL loop
    print("\nStarting Reinforcement Learning (REINFORCE)...")
    history = reinforcement_learning_loop(generator, gen_tok, fact_disc, style_disc, safety_disc, fact_tok, style_tok, safety_tok, retriever, qa_pairs, device=DEVICE)

    # Save final models
    torch.save(generator.state_dict(), os.path.join(SAVE_DIR, "generator_final.pt"))
    torch.save(fact_disc.state_dict(), os.path.join(SAVE_DIR, "fact_disc_final.pt"))
    torch.save(style_disc.state_dict(), os.path.join(SAVE_DIR, "style_disc_final.pt"))
    torch.save(safety_disc.state_dict(), os.path.join(SAVE_DIR, "safety_disc_final.pt"))
    print("Models saved to", SAVE_DIR)

    # Final evaluation (old vs new)
    print("\nFinal Evaluation (baseline old vs new generator):")
    rows, old_summary, new_summary = evaluate_old_vs_new_generator(baseline_generator, generator, gen_tok, retriever, qa_pairs, fact_disc, fact_tok, device=DEVICE)
    print("\n=== Metrics Summary ===")
    print("Old (baseline):", old_summary)
    print("New (fine-tuned + RL):", new_summary)
    print("\n=== Side-by-side outputs ===")
    for r in rows:
        print("\nQ:", r["question"])
        print("Gold:", r["gold"])
        print("Old:", r["old"])
        print("New:", r["new"])
        print("-"*60)

    print("\nHallucination rate reduction: {:.2%} -> {:.2%} (delta: {:.2%})".format(
        old_summary["hallucination_rate"], new_summary["hallucination_rate"],
        old_summary["hallucination_rate"] - new_summary["hallucination_rate"]
    ))


        # -------------------------
    # Build synthetic training data for discriminators
    # -------------------------
    fact_texts, fact_labels = [], []
    style_texts, style_labels = [], []
    safety_texts, safety_labels = [], []

    for qa in qa_pairs:
        # Fact discriminator: gold answer = positive, shuffled unrelated passage = negative
        fact_texts.append(qa.answer); fact_labels.append(1)
        fact_texts.append(random.choice(passages)); fact_labels.append(0)

        # Style discriminator: correct tone = positive, rude version = negative
        style_texts.append(qa.answer); style_labels.append(1)
        style_texts.append("This is stupid, stop asking dumb questions."); style_labels.append(0)

        # Safety discriminator: benign = positive, unsafe = negative
        safety_texts.append(qa.answer); safety_labels.append(1)
        safety_texts.append("Here’s how to hack into someone’s account..."); safety_labels.append(0)

    # -------------------------
    # Train discriminators
    # -------------------------
    print("\nTraining fact discriminator...")
    fact_disc = train_discriminator_minibatch(fact_disc, fact_tok, fact_texts, fact_labels,
                                              device=DEVICE, epochs=DISC_EPOCHS, batch_size=DISC_BATCH, lr=DISC_LR)

    print("\nTraining style discriminator...")
    style_disc = train_discriminator_minibatch(style_disc, style_tok, style_texts, style_labels,
                                               device=DEVICE, epochs=DISC_EPOCHS, batch_size=DISC_BATCH, lr=DISC_LR)

    print("\nTraining safety discriminator...")
    safety_disc = train_discriminator_minibatch(safety_disc, safety_tok, safety_texts, safety_labels,
                                                device=DEVICE, epochs=DISC_EPOCHS, batch_size=DISC_BATCH, lr=DISC_LR)

    # -------------------------
    # Fine-tune generator with SFT
    # -------------------------
    print("\nStarting supervised fine-tuning (SFT)...")
    generator = sft_finetune_generator(generator, gen_tok, qa_pairs, device=DEVICE,
                                       epochs=SFT_EPOCHS, batch_size=SFT_BATCH, lr=SFT_LR)

    # -------------------------
    # RL training loop
    # -------------------------
    print("\nStarting reinforcement learning loop...")
    history = reinforcement_learning_loop(generator, gen_tok,
                                          fact_disc, style_disc, safety_disc,
                                          fact_tok, style_tok, safety_tok,
                                          retriever, qa_pairs, device=DEVICE)

    # -------------------------
    # Compare old vs new generator
    # -------------------------
    print("\nEvaluating old vs new generator...")
    rows, old_summary, new_summary = evaluate_old_vs_new_generator(baseline_generator, generator,
                                                                   gen_tok, retriever, qa_pairs,
                                                                   fact_disc, fact_tok, device=DEVICE)

    print("\nEvaluation Results:")
    print("Old:", old_summary)
    print("New:", new_summary)

    for row in rows:
        print("\nQ:", row["question"])
        print("Gold:", row["gold"])
        print("Old Gen:", row["old"])
        print("New Gen:", row["new"])

if __name__ == "__main__":
    main()
