import os
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import load_config, QADataset


def main():
    # ── 1. Load config ────────────────────────────────────────────
    config = load_config("configs/config.yaml")
    print("Config loaded:", config)

    # ── 2. Load tokenizer + model ─────────────────────────────────
    print(f"\nLoading model: {config['model_name']} ...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # GPT-2 has no pad token by default — we set it to eos_token
    # Without this, padding will crash during tokenization
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    print(f"Base model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 3. Inject LoRA ────────────────────────────────────────────
    # LoraConfig defines WHERE and HOW we inject trainable matrices
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,       # we're doing causal language modeling
        r=config["lora_r"],                 # rank of the low-rank matrices
        lora_alpha=config["lora_alpha"],    # scaling factor
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],  # which layers to inject into
    )

    model = get_peft_model(model, lora_config)

    # This shows you exactly how few parameters LoRA trains vs the full model
    model.print_trainable_parameters()

    # ── 4. Load dataset ───────────────────────────────────────────
    print("\nLoading dataset...")
    dataset = QADataset(
        data_path=config["data_path"],
        tokenizer=tokenizer,
        max_length=config["max_length"],
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    print(f"Dataset size: {len(dataset)} examples")

    # ── 5. Optimizer ──────────────────────────────────────────────
    # Only LoRA parameters get updated — frozen base model params are skipped
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
    )

    # ── 6. Training loop ──────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on: {device}")
    model.to(device)
    model.train()

    for epoch in range(config["num_epochs"]):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # Backward pass — compute gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} complete. Avg loss: {avg_loss:.4f}")

    # ── 7. Save LoRA weights ──────────────────────────────────────
    os.makedirs(config["output_dir"], exist_ok=True)
    model.save_pretrained(config["output_dir"])
    print(f"\nLoRA weights saved to {config['output_dir']}")


if __name__ == "__main__":
    main()