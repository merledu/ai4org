from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer
from config import DEFAULT_MODEL
from transformers import BitsAndBytesConfig



def load_model_tokenizer(model_name=DEFAULT_MODEL):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model.config.pad_token_id = tok.eos_token_id
    return tok, model
