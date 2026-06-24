from config_reader import load_config

cfg = load_config("config/pipeline_config.yaml")
MAX_Q_PER_CHUNK = cfg.get("max_q_per_chunk", 3)


def build_prompt(passage: str, max_q: int = MAX_Q_PER_CHUNK) -> str:
    PROMPT_TEMPLATE = f"""
  You are generating high-quality question–answer pairs for fine-tuning a language model to reduce hallucinations.

  Follow these strict rules:

  Question Guidelines
  Questions must sound natural and realistic, as if asked by a bank customer, employee, or compliance officer.
  Do NOT mention section numbers, clause numbers, or internal policy labels.
  Questions must be specific and unambiguous.
  Avoid generic phrases like “What does the policy say”.

  Questions must be fully self-contained and standalone.
  Do NOT use vague references such as:
  - "this policy"
  - "this context"
  - "this document"
  - "this section"
  - "the above"
  - "here"
  Instead, explicitly name the policy, entity, or subject described in the text.
  Canonical Subject Rule:
  Each question MUST explicitly name the policy, process, or entity described in the passage (e.g., “Amanah Bank’s Account Opening and Maintenance Policy”).
  Generic references such as “the policy” are NOT allowed.

  Answer Guidelines
  Answers must be fully supported by the provided text.
  Answers must be complete, factual, and standalone.
  Do NOT repeat the question wording.
  Do NOT add assumptions or external knowledge.
  Answers should be concise and limited to the facts explicitly stated in the passage.

  Grounding Requirement
  Each answer must be directly traceable to one or more exact sentences in the text.
  Do not combine information from unrelated sections.
  Generate up to {max_q} high-quality question–answer pairs from the text. Generating zero questions is acceptable.

  Generation Preference
  If the passage contains clear rules, responsibilities, timelines, or conditions, generate at least one question.

  Negative instruction
  If the text does not clearly answer a potential question, do NOT generate that question.

  Output ONLY the Q&A pairs in the exact format shown. Do not include explanations or commentary.

  Format:
  Q1: <question>?
  A1: <answer>

  Q2: <question>?
  A2: <answer>

  ... up to Q{max_q}

    Passage:
    {passage}
  """
    return PROMPT_TEMPLATE.format(max_q=max_q, passage=passage)
