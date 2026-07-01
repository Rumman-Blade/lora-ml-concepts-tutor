import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils import load_config


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 100, device: str = "cpu") -> str:
    """
    Generate an answer from the model given a question.
    
    We format the prompt the same way we did during training
    so the model recognizes the pattern it learned.
    """
    prompt = f"### Question: {question}\n### Answer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():  # no gradients needed for inference
        outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
    pad_token_id=tokenizer.eos_token_id,
)
    
    # Decode only the newly generated tokens (not the prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    config = load_config("configs/config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running eval on: {device}\n")

    # ── Load tokenizer ────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    # ── Load base model (no LoRA) ─────────────────────────────────
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    base_model.to(device)
    base_model.eval()

    # ── Load LoRA fine-tuned model ────────────────────────────────
    print("Loading LoRA fine-tuned model...")
    lora_model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    lora_model = PeftModel.from_pretrained(lora_model, config["output_dir"])
    lora_model.to(device)
    lora_model.eval()

    # ── Eval questions ────────────────────────────────────────────
    questions = [
        "What is a neural network?",
        "What is gradient descent?",
        "What is overfitting?",
        "What is LoRA?",
        "What is fine-tuning?",
    ]

    print("\n" + "="*70)
    print("EVALUATION: Base GPT-2 vs LoRA Fine-tuned GPT-2")
    print("="*70)

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}] Question: {question}")
        print("-" * 50)

        base_answer = generate_answer(base_model, tokenizer, question, device=device)
        print(f"Base GPT-2:\n{base_answer}")
        print()

        lora_answer = generate_answer(lora_model, tokenizer, question, device=device)
        print(f"LoRA Model:\n{lora_answer}")
        print("="*70)


if __name__ == "__main__":
    main()