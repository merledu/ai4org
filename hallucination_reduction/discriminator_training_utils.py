# hallucination_reduction/discriminator_training_utils.py

from discriminator import train_discriminator_minibatch

def train_all_discriminators(
    fact_disc, fact_tok,
    style_disc, style_tok,
    safety_disc, safety_tok,
    qa_pairs,
    device, epochs, batch_size, lr
):
    """
    Train fact, style, and safety discriminators with both positive and negative examples.
    Positive examples come from real QA answers.
    Negative examples are synthetic dummy texts.
    """

    # --- Fact discriminator dataset ---
    fact_texts, fact_labels = [], []
    for qa in qa_pairs:
        fact_texts.append(qa.answer)  # Positive (real answer)
        fact_labels.append(1)
        fact_texts.append("This is unrelated nonsense.")  # Negative
        fact_labels.append(0)

    # --- Style discriminator dataset ---
    style_texts, style_labels = [], []
    for qa in qa_pairs:
        style_texts.append(qa.answer)  # Positive
        style_labels.append(1)
        style_texts.append("ugly broken text without punctuation or coherence")  # Negative
        style_labels.append(0)

    # --- Safety discriminator dataset ---
    safety_texts, safety_labels = [], []
    for qa in qa_pairs:
        safety_texts.append(qa.answer)  # Positive
        safety_labels.append(1)
        safety_texts.append("unsafe / harmful / toxic content")  # Negative
        safety_labels.append(0)

    # --- Train each discriminator ---
    fact_disc = train_discriminator_minibatch(
        fact_disc, fact_tok, fact_texts, fact_labels,
        epochs=epochs, batch_size=batch_size, lr=lr, device=device
    )

    style_disc = train_discriminator_minibatch(
        style_disc, style_tok, style_texts, style_labels,
        epochs=epochs, batch_size=batch_size, lr=lr, device=device
    )

    safety_disc = train_discriminator_minibatch(
        safety_disc, safety_tok, safety_texts, safety_labels,
        epochs=epochs, batch_size=batch_size, lr=lr, device=device
    )

    return fact_disc, style_disc, safety_disc
