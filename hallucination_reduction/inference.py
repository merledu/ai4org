import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(base_model_name=None, lora_path=None, device=None):
    """
    Loads a base model and optionally merges LoRA weights (.pt or folder).
    Works with GPT-2 or any Hugging Face model.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- Auto-detect GPT-2 if LoRA was trained on it ---
    if base_model_name is None:
        if lora_path and "gpt2" in lora_path.lower():
            base_model_name = "gpt2"
        else:
            base_model_name = "gpt2"  # default fallback

    print(f"🔧 Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # --- Load LoRA if provided ---
    if lora_path:
        if os.path.isdir(lora_path):
            print(f"🔄 Loading LoRA adapter from folder: {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)
        elif os.path.isfile(lora_path) and lora_path.endswith(".pt"):
            print(f"🔄 Loading LoRA weights manually from: {lora_path}")
            state_dict = torch.load(lora_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded LoRA weights — Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        else:
            print(f"⚠️ Invalid LoRA path: {lora_path}")

    model.eval()
    return model, tokenizer, device


def chat(model, tokenizer, device):
    print("\n💬 LoRA + GPT-2 Chat Mode — type 'exit' to quit.\n")

    while True:
        user_input = input("❓ You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Exiting chat.")
            break

        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n🤖 Bot: {response}\n")


def main():
    # 🔧 Change these paths if needed
    base_model = "gpt2"  # since LoRA was trained on GPT-2
    lora_path = "./saved_models_improved/generator_final.pt"  # can be .pt or folder

    model, tokenizer, device = load_model(base_model, lora_path)
    chat(model, tokenizer, device)


if __name__ == "__main__":
    main()
