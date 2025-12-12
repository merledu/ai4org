from config import  MAX_Q_PER_CHUNK


def build_prompt(passage: str, max_q: int=MAX_Q_PER_CHUNK) -> str:
    PROMPT_TEMPLATE = f"""
    You are an expert banking compliance analyst. From the passage below, produce between 0 and {max_q} very specific Q&A pairs that refer *explicitly* to policy names or section/clause numbers present in the passage.

    REQUIREMENTS:
    - Generate ONLY questions that include a policy name or a section/clause number (e.g., "Section 2.3 Customer Risk Assessment and Profiling Policy" or "Section 5.1 Savings Account Interest Policy").
    - Do NOT produce vague questions (e.g., "What is the purpose of this policy?") unless the policy name or number is explicitly mentioned in the question.
    - Each answer must be 1–3 sentences, directly supported by the passage (do not invent facts).
    - Output EXACTLY in this format (no extra commentary):

    Q1: <question>?
    A1: <answer>

    Q2: <question>?
    A2: <answer>

    ... up to Q{max_q}

    Passage:
    {passage}
    """
    return PROMPT_TEMPLATE.format(max_q=max_q, passage=passage)
