"""
scripts/finetune_lora.py — Level 3: QLoRA Fine-Tuning with Unsloth
====================================================================
Fine-tunes a LLaMA model on manufacturing data using QLoRA (Quantized LoRA).

Fine-tuning Hierarchy:
  Full Fine-Tuning  → updates all 7B+ parameters  → needs 80–140 GB VRAM
  LoRA              → trains ~0.4% (28M) params    → needs 24–40 GB VRAM
  QLoRA (THIS FILE) → trains ~0.4% params + 4-bit  → needs 6–12 GB VRAM ✅

Key Concepts:
  PEFT   — Parameter-Efficient Fine-Tuning: umbrella term for low-cost tuning methods
  LoRA   — Low-Rank Adaptation: adds trainable A×B matrices alongside frozen weights
             W' = W + (alpha/rank) × B × A
  QLoRA  — Quantized LoRA: base model loaded in 4-bit NF4 (via bitsandbytes),
             LoRA adapters trained in bfloat16. Typical consumer GPU approach.

Requirements:
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install bitsandbytes datasets trl

Usage:
  python scripts/prepare_data.py          # export to Alpaca format first
  python scripts/finetune_lora.py         # run QLoRA training

Output:
  outputs/mfg-qlora-adapter/             — saved LoRA adapters (SafeTensors)
  outputs/mfg-qlora.Q4_K_M.gguf         — merged GGUF for Ollama import

Import into Ollama after training:
  ollama create mfg-expert-qlora -f modelfiles/Modelfile.qlora
"""

import json
import os
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_MODEL        = "unsloth/llama-3.2-3b-bnb-4bit"   # 4-bit quantized base
ALPACA_DATA_PATH  = Path("data/manufacturing_train_alpaca.json")
OUTPUT_DIR        = Path("outputs")
ADAPTER_DIR       = OUTPUT_DIR / "mfg-qlora-adapter"
GGUF_NAME         = "mfg-qlora"

# ── LoRA Hyperparameters ───────────────────────────────────────────────────────
# rank (r): controls adapter expressiveness. Higher = richer but more params.
#   r=4  → 3.3M params  — minimal, fast, slight quality loss
#   r=16 → 13M params   — good balance for domain-specific niche vocab ✅
#   r=64 → 52M params   — near full fine-tune quality, more VRAM
LORA_RANK         = 16

# alpha (α): scaling factor for LoRA output. Start with alpha = rank.
#   Higher alpha = stronger LoRA influence, risk of overfitting on small data.
LORA_ALPHA        = 16

# dropout: regularization for LoRA layers. 0 = no dropout (recommended for small data).
LORA_DROPOUT      = 0.05

# target_modules: which transformer projection layers get LoRA adapters.
#   Using all 7 layers gives best domain adaptation coverage for LLaMA.
LORA_TARGET_MODULES = [
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training Hyperparameters ───────────────────────────────────────────────────
MAX_SEQ_LENGTH    = 2048   # tokens per sample
NUM_TRAIN_EPOCHS  = 3      # 3 passes through the dataset (increase to 5 for very small datasets)
PER_DEVICE_BATCH  = 2      # reduce to 1 if OOM
GRADIENT_ACCUM    = 4      # effective batch = PER_DEVICE_BATCH × GRADIENT_ACCUM = 8
LEARNING_RATE     = 2e-4   # AdamW default; lower (1e-4) for small datasets to avoid overfitting
WARMUP_STEPS      = 10
LOGGING_STEPS     = 5
SAVE_STEPS        = 50
SEED              = 42


def load_alpaca_data(path: Path) -> list[dict]:
    """Load the Alpaca-format training data exported by prepare_data.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found: {path}\n"
            "Run:  python scripts/prepare_data.py"
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} training samples from {path}")
    return data


def format_alpaca_prompt(instruction: str, input_text: str, output: str = "") -> str:
    """
    Format a sample as an Alpaca prompt.
    Unsloth expects a single string; the EOS token is appended during training.
    """
    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
    return prompt


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── 1. Load Model + Tokenizer (QLoRA: 4-bit NF4 quantization) ─────────────
    print(f"\n[1/5] Loading base model: {BASE_MODEL}")
    print("      Base model is quantized to 4-bit NF4 (QLoRA approach).")
    print("      VRAM footprint for 3B model: ~3–4 GB | 7B model: ~6–8 GB\n")

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed.")
        print("Run:  pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
        raise

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,          # QLoRA: 4-bit NF4 base
        dtype=None,                 # auto-detect (bfloat16 on Ampere+)
    )

    # ── 2. Add LoRA Adapters (PEFT) ────────────────────────────────────────────
    print(f"[2/5] Attaching LoRA adapters (PEFT)")
    print(f"      rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"      target_modules: {LORA_TARGET_MODULES}\n")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",              # do not train bias terms
        use_gradient_checkpointing="unsloth",  # saves ~30% VRAM
        random_state=SEED,
    )

    # Print trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"      Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"      (This is the 'P' in PEFT — only {100*trainable/total:.2f}% of parameters updated)\n")

    # ── 3. Prepare Dataset ─────────────────────────────────────────────────────
    print("[3/5] Preparing manufacturing training dataset")

    try:
        from datasets import Dataset
    except ImportError:
        print("ERROR: datasets not installed.  Run:  pip install datasets")
        raise

    samples = load_alpaca_data(ALPACA_DATA_PATH)

    def tokenize(batch):
        prompts = [
            format_alpaca_prompt(
                s.get("instruction", ""),
                s.get("input", ""),
                s.get("output", ""),
            )
            for s in batch
        ]
        return tokenizer(prompts, truncation=True, max_length=MAX_SEQ_LENGTH)

    dataset = Dataset.from_list(samples)
    print(f"      Dataset: {len(dataset)} samples\n")

    # ── 4. Train ───────────────────────────────────────────────────────────────
    print("[4/5] Starting QLoRA training")

    try:
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError:
        print("ERROR: trl/transformers not installed.  Run:  pip install trl transformers")
        raise

    alpaca_prompt_template = (
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="output",       # SFTTrainer formats via formatting_func
        formatting_func=lambda examples: [
            format_alpaca_prompt(
                examples["instruction"][i] if isinstance(examples["instruction"], list) else examples["instruction"],
                examples.get("input", [""] * len(examples["instruction"]))[i] if isinstance(examples.get("input", [""]), list) else examples.get("input", ""),
                examples["output"][i] if isinstance(examples["output"], list) else examples["output"],
            )
            for i in range(len(examples["instruction"]) if isinstance(examples["instruction"], list) else 1)
        ],
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            output_dir=str(ADAPTER_DIR),
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRADIENT_ACCUM,
            warmup_steps=WARMUP_STEPS,
            learning_rate=LEARNING_RATE,
            fp16=not _is_bf16_available(),
            bf16=_is_bf16_available(),
            logging_steps=LOGGING_STEPS,
            save_steps=SAVE_STEPS,
            save_total_limit=2,
            optim="adamw_8bit",    # 8-bit AdamW reduces optimizer VRAM usage
            seed=SEED,
            report_to="none",
        ),
    )

    print(f"      epochs={NUM_TRAIN_EPOCHS}, batch={PER_DEVICE_BATCH}×{GRADIENT_ACCUM}, lr={LEARNING_RATE}")
    trainer_stats = trainer.train()

    elapsed = trainer_stats.metrics.get("train_runtime", 0)
    print(f"\n      Training complete! {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── 5. Save Adapters + Export GGUF ─────────────────────────────────────────
    print(f"\n[5/5] Saving LoRA adapters → {ADAPTER_DIR}")
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    print(f"      Exporting merged GGUF → {OUTPUT_DIR}/{GGUF_NAME}.Q4_K_M.gguf")
    print("      (Q4_K_M quantization: best quality/size tradeoff for Ollama)\n")

    model.save_pretrained_gguf(
        str(OUTPUT_DIR / GGUF_NAME),
        tokenizer,
        quantization_method="q4_k_m",
    )

    # Write Modelfile for Ollama import
    modelfile_path = Path("modelfiles/Modelfile.qlora")
    modelfile_path.parent.mkdir(exist_ok=True)
    modelfile_path.write_text(
        f'FROM ./{OUTPUT_DIR}/{GGUF_NAME}.Q4_K_M.gguf\n\n'
        'PARAMETER temperature 0.25\n'
        'PARAMETER num_ctx 8192\n'
        'PARAMETER repeat_penalty 1.1\n\n'
        'SYSTEM """You are ManuBot, a manufacturing expert AI fine-tuned with QLoRA '
        'on domain-specific data. Provide precise, structured answers citing relevant '
        'ISO standards, OEE metrics, and maintenance procedures."""\n',
        encoding="utf-8",
    )

    print("=" * 60)
    print("SUCCESS! Import your QLoRA model into Ollama:")
    print()
    print(f"  ollama create mfg-expert-qlora -f {modelfile_path}")
    print()
    print("Then open the Comparator, select mfg-expert-qlora,")
    print("and compare it against mfg-base and mfg-expert side-by-side!")
    print("=" * 60)


def _is_bf16_available() -> bool:
    """Check if bfloat16 is supported (Ampere+ GPU)."""
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
