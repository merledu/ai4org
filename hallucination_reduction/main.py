from .data_utils import build_corpus_and_qa
from .retriever import SimpleRetriever
from .generator import load_generator, sft_finetune_generator
from .discriminator import load_discriminator
from .discriminator_training_utils import train_discriminator_minibatch, evaluate_classifier
from .evaluation import evaluate_old_vs_new_generator
from .rl_utils import reinforcement_learning_loop
from .config import GEN_MODEL, DISC_MODEL, SAVE_DIR, SFT_EPOCHS, SFT_BATCH, SFT_LR, DISC_EPOCHS, DISC_BATCH, DISC_LR, RL_EPOCHS
import copy
import torch
import os
import random
import numpy as np
from torch import nn



def unwrap(model):
    """Return the original model if wrapped in DataParallel."""
    return model.module if hasattr(model, "module") else model


def main():
    
    if torch.cuda.is_available():
        DEVICE = "cuda"  
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"  
    else:
        DEVICE = "cpu"
    print(f" Using device: {DEVICE}")

    
    passages, qa_pairs = build_corpus_and_qa()
    retriever = SimpleRetriever(passages)

    
    print("Loading generator and discriminators...")
    # Load generator to CPU first to save memory
    gen_tok, generator = load_generator(GEN_MODEL, "cpu")
    baseline_generator = copy.deepcopy(generator) # Keep on CPU

    # Load discriminators to CPU first
    fact_tok, fact_disc = load_discriminator(DISC_MODEL, "cpu")
    style_tok, style_disc = load_discriminator(DISC_MODEL, "cpu")
    safety_tok, safety_disc = load_discriminator(DISC_MODEL, "cpu")

    
    try:
        if hasattr(generator, "gradient_checkpointing_enable"):
            generator.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing on generator.")
    except Exception as e:
        print("Could not enable gradient checkpointing:", e)

    
    # We will move models to DEVICE only when needed
    print(f"Main computation device: {DEVICE}")

    # DataParallel wrapping for generator (if applicable)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 1:
        print(f" Using {n_gpus} GPUs with DataParallel (generator only).")
        # DataParallel moves the model to GPU automatically
        generator = nn.DataParallel(generator)
    else:
        print("Using single GPU or CPU.")


    fact_texts, fact_labels = [], []
    style_texts, style_labels = [], []
    safety_texts, safety_labels = [], []

    fabrication_patterns = [
        " Additionally, we share all user records publicly for transparency.",
        " Moreover, we store sensitive data indefinitely on third-party servers.",
        " We also allow any employee to view confidential customer information.",
        " Furthermore, our policy permits external contractors to access logs.",
        " We retain personal records permanently, even after user deletion requests.",
        " The organization openly publishes transaction logs for analysis.",
    ]

    unsafe_patterns = [
        " We freely distribute private user data to marketing partners.",
        " Employees can access any confidential data at their discretion.",
        " We don’t require consent before processing personal information.",
        " Sensitive information is shared with affiliates for operational efficiency.",
    ]

    bad_style_patterns = [
        " bruh, just follow the rules lol.",
        " idk man, seems fine to me.",
        " whatever, we’ll see what happens.",
        " nah bro, do what you want.",
        " who cares, it’s all good.",
    ]

    for qa in qa_pairs:
        
        fact_texts.append(qa.answer); fact_labels.append(1)
        style_texts.append(qa.answer); style_labels.append(1)
        safety_texts.append(qa.answer); safety_labels.append(1)

        
        fabricated = qa.answer + random.choice(fabrication_patterns)
        fact_texts.append(fabricated); fact_labels.append(0)
        safety_texts.append(fabricated); safety_labels.append(0)

        
        unsafe = qa.answer + random.choice(unsafe_patterns)
        fact_texts.append(unsafe); fact_labels.append(0)
        safety_texts.append(unsafe); safety_labels.append(0)

        
        bad_style = random.choice(bad_style_patterns)
        style_texts.append(bad_style); style_labels.append(0)

        
        random_neg = "Misc unrelated content " + str(random.randint(0, 10000))
        fact_texts.append(random_neg); fact_labels.append(0)
        style_texts.append(random_neg); style_labels.append(0)
        safety_texts.append(random_neg); safety_labels.append(0)

    
    
    print("Training fact discriminator...")
    # Move to GPU for training, then back to CPU
    fact_disc = fact_disc.to(DEVICE)
    fact_disc = train_discriminator_minibatch(fact_disc, fact_tok, fact_texts, fact_labels,
                                              device=DEVICE, epochs=DISC_EPOCHS, batch_size=DISC_BATCH, lr=DISC_LR)
    fact_disc = fact_disc.to("cpu")
    torch.cuda.empty_cache()

    print("Training style discriminator...")
    style_disc = style_disc.to(DEVICE)
    style_disc = train_discriminator_minibatch(style_disc, style_tok, style_texts, style_labels,
                                               device=DEVICE, epochs=DISC_EPOCHS, batch_size=DISC_BATCH, lr=DISC_LR)
    style_disc = style_disc.to("cpu")
    torch.cuda.empty_cache()

    print("Training safety discriminator...")
    safety_disc = safety_disc.to(DEVICE)
    safety_disc = train_discriminator_minibatch(safety_disc, safety_tok, safety_texts, safety_labels,
                                                device=DEVICE, epochs=DISC_EPOCHS, batch_size=DISC_BATCH, lr=DISC_LR)
    safety_disc = safety_disc.to("cpu")
    torch.cuda.empty_cache()

    

    print("\nSanity-check discriminator metrics:")
    # Move discriminators to DEVICE for evaluation
    fact_disc = fact_disc.to(DEVICE)
    style_disc = style_disc.to(DEVICE)
    safety_disc = safety_disc.to(DEVICE)
    print("Fact disc:", evaluate_classifier(fact_disc, fact_tok, fact_texts, fact_labels, device=DEVICE))
    print("Style disc:", evaluate_classifier(style_disc, style_tok, style_texts, style_labels, device=DEVICE))
    print("Safety disc:", evaluate_classifier(safety_disc, safety_tok, safety_texts, safety_labels, device=DEVICE))

    

    print("\nEvaluation before SFT/RL (baseline):")
    # Move generator to GPU for evaluation
    generator = generator.to(DEVICE)
    # Move discriminators to CPU to save space if not needed, but here we need them for metrics?
    # evaluate_old_vs_new_generator uses fact_disc for hallucination check
    # fact_disc is already on DEVICE from previous block
    
    _, old_before, _ = evaluate_old_vs_new_generator(baseline_generator, generator,
                                                     gen_tok, retriever, qa_pairs,
                                                     fact_disc, fact_tok, device=DEVICE)
    print("Baseline summary:", old_before)

    

    print("\nRunning supervised fine-tuning (SFT) on QA pairs...")
    # Generator is already on DEVICE
    # We can move discriminators to CPU to save space during SFT
    fact_disc = fact_disc.to("cpu")
    style_disc = style_disc.to("cpu")
    safety_disc = safety_disc.to("cpu")
    torch.cuda.empty_cache()
    
    generator = sft_finetune_generator(generator, gen_tok, qa_pairs,
                                       device=DEVICE, epochs=SFT_EPOCHS,
                                       batch_size=SFT_BATCH, lr=SFT_LR)

    print("\nEvaluation after SFT (before RL):")
    # Need fact_disc on DEVICE for evaluation
    fact_disc = fact_disc.to(DEVICE)
    
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

    

    print("\nStarting Reinforcement Learning (REINFORCE)...")
    # Move all discriminators to DEVICE for RL
    fact_disc = fact_disc.to(DEVICE)
    style_disc = style_disc.to(DEVICE)
    safety_disc = safety_disc.to(DEVICE)
    
    history = reinforcement_learning_loop(generator, gen_tok,
                                          fact_disc, style_disc, safety_disc,
                                          fact_tok, style_tok, safety_tok,
                                          retriever, qa_pairs, device=DEVICE)

    

    torch.save(unwrap(generator).state_dict(), os.path.join(SAVE_DIR, "generator_final.pt"))
    torch.save(unwrap(fact_disc).state_dict(), os.path.join(SAVE_DIR, "fact_disc_final.pt"))
    torch.save(unwrap(style_disc).state_dict(), os.path.join(SAVE_DIR, "style_disc_final.pt"))
    torch.save(unwrap(safety_disc).state_dict(), os.path.join(SAVE_DIR, "safety_disc_final.pt"))
    print("✅ Models saved to", SAVE_DIR)

    
    
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
