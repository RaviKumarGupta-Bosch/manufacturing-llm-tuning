# Step-by-Step Guide: Fine-Tuning Ollama Models for Manufacturing

## Introduction

This tutorial walks you through customizing a local LLM (via Ollama) for the **manufacturing domain**, then running a live side-by-side comparison of the base model vs. the tuned model.

---

## What Is Model Tuning?

When you pull a model like `llama3.2` from Ollama, it has broad knowledge but no specialization. It will give generic answers. **Tuning** reshapes the model's behavior toward your domain — giving it vocabulary, response structure, domain thresholds (OEE > 85% = world class), and preferred formats.

There are three levels of tuning in this project:

```
Level 1 — Modelfile (Minutes)
  System prompt + few-shot examples + parameter tuning
  No GPU needed, works immediately

Level 2 — RAG (Hours)
  Retrieval-Augmented Generation with a knowledge base
  No weight changes, dynamic domain grounding

Level 3 — LoRA Fine-Tuning (Days)
  Actual gradient-based weight updates
  Requires GPU (NVIDIA 8GB+ VRAM recommended)
  Export to GGUF → import back to Ollama
```

---

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| Ollama | Local LLM runtime | https://ollama.com/download |
| Python 3.9+ | Scripts and UI backend | https://python.org |
| pip packages | FastAPI, httpx, etc. | `pip install -r ui/requirements.txt` |
| llama3.2 model | Base model | `ollama pull llama3.2` |

---

## Step 1 — Understand the Project Structure

```
Model-Tuning/
├── data/
│   ├── manufacturing_train.jsonl   ← 10 expert Q&A conversations
│   ├── manufacturing_test.jsonl    ← 8 evaluation test cases
│   └── knowledge_base.md          ← Domain reference (OEE, FMEA, standards)
├── modelfiles/
│   ├── Modelfile.base              ← "Before" — minimal system prompt
│   └── Modelfile.manufacturing     ← "After" — domain-tuned configuration
├── scripts/
│   ├── setup_env.py                ← Check dependencies
│   ├── prepare_data.py             ← Validate & export training data
│   ├── create_models.py            ← Register models with Ollama
│   └── evaluate_models.py          ← Score both models on test set
├── ui/
│   ├── app.py                      ← FastAPI backend
│   └── static/                     ← Side-by-side HTML/CSS/JS UI
└── tests/
    └── test_cases.py               ← Pytest unit tests
```

---

## Step 2 — Check Your Environment

```bash
python scripts/setup_env.py
```

Expected output:
```
[OK]   Python 3.11.x
[OK]   Ollama 0.x.x
[OK]   Ollama API running. Models available: 3
[OK]   Dependencies installed successfully
[READY] Environment is set up.
```

If you see `[WARN] Ollama API not responding`, run `ollama serve` in a separate terminal and retry.

---

## Step 3 — Understand the Training Data Format

Open `data/manufacturing_train.jsonl`. Each line is one training conversation in **ChatML format** (messages array):

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are ManuBot, a senior manufacturing engineer AI..."
    },
    {
      "role": "user",
      "content": "CNC spindle vibration jumped from 0.8 to 3.9 mm/s. What should I do?"
    },
    {
      "role": "assistant",
      "content": "**Fault Analysis: Spindle Vibration...\n\nRoot Cause:..."
    }
  ]
}
```

**Why this format matters:**
- The **system** message sets the persona and constraints
- The **user/assistant** pairs are examples of the expected behavior
- The quality of these examples determines the quality of the tuning

### Validate your data

```bash
python scripts/prepare_data.py
```

This will:
1. Check every record for valid JSON and required fields
2. Report token counts and average response lengths
3. Export to `data/manufacturing_train_alpaca.json` (needed for Level 3 LoRA)

---

## Step 4 — Level 1 Tuning: Modelfile Engineering

The simplest and fastest approach. Open `modelfiles/Modelfile.manufacturing`:

```dockerfile
FROM llama3.2

PARAMETER temperature 0.3      # Lower = more focused, deterministic
PARAMETER top_p 0.85
PARAMETER num_ctx 8192          # Larger context for long structured responses
PARAMETER repeat_penalty 1.1   # Reduce repetition

SYSTEM """You are ManuBot, a senior manufacturing engineer AI with 20+ years...
...Always respond with: Root Cause, Recommended Action, Priority Level, Standard."""

MESSAGE user "CNC spindle vibration jumped..."
MESSAGE assistant "**SAFETY NOTE:** ..."
```

**What each part does:**

| Directive | Effect |
|-----------|--------|
| `FROM llama3.2` | Which base model weights to use |
| `PARAMETER temperature 0.3` | Reduces randomness — critical for technical accuracy |
| `SYSTEM "..."` | Baked-in persona and response format instructions |
| `MESSAGE user/assistant` | Few-shot examples that shape output style |

### Create the models

```bash
python scripts/create_models.py
```

This registers two models with Ollama:
- `mfg-base` — the baseline (minimal prompt)
- `mfg-expert` — the manufacturing-tuned model

Verify with:
```bash
ollama list
```

You should see `mfg-base` and `mfg-expert` in the list.

---

## Step 5 — Run the Side-by-Side UI

```bash
cd ui
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

### UI Walkthrough

```
┌─────────────────────────────────────────────────────────────┐
│  Manufacturing LLM Tuning      [● Ollama Online]            │
├──────────────────────────────────────────────────────────────┤
│  Left Model:  [mfg-base ▼] BASE     VS  [mfg-expert ▼] TUNED│
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ CNC spindle vibration jumped from 0.8 to 3.9 mm/s.  │   │
│  │ What should I do?                          [Compare ▶]│  │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────┬────────────────────────────────────────┤
│  BASE MODEL         │  TUNED MODEL                          │
│  (No domain ctx)    │  (Manufacturing Expert)               │
│  ─────────────────  │  ─────────────────────────────────── │
│  "Increased         │  **SAFETY NOTE:** At 3.9 mm/s...     │
│  vibration can      │                                       │
│  indicate issues    │  **Root Cause / Analysis:**          │
│  with bearings..."  │  1. Bearing inner race crack          │
│                     │  2. Coolant contamination             │
│                     │                                       │
│                     │  **Recommended Action:**              │
│                     │  1. Reduce speed 30%...               │
│                     │                                       │
│                     │  **Priority:** HIGH                   │
│                     │  **Standard:** ISO 10816-3            │
│  2.1s  3.2 t/s ⎘  │  3.4s  2.9 t/s  ⎘                   │
└─────────────────────┴───────────────────────────────────────┘
│ Sample Prompts: [OEE] [FMEA] [Safety] [Robot Fault] ...     │
└─────────────────────────────────────────────────────────────┘
```

**Try the sample prompts** (click "Sample Prompts" button) to quickly test all 10 manufacturing categories.

---

## Step 6 — Run Automated Evaluation

```bash
python scripts/evaluate_models.py
```

This runs all 8 test cases against both models and produces a scored comparison:

```
═══════════════════════════════════════════════════════════════════════════
  EVALUATION RESULTS
═══════════════════════════════════════════════════════════════════════════
  Test ID  Category                       Base Score  Expert Score  Delta
  ───────  ───────────────────────────── ──────────  ────────────  ──────
  T001     predictive_maintenance               41%          87%    +46
  T002     quality_defect                       38%          84%    +46
  T003     oee_analysis                         52%          91%    +39
  T004     safety                               45%          88%    +43
  T005     fmea                                 35%          82%    +47
  ───────  ─────────────────────────────────────────────────────────────
  AVERAGE                                       42%          86%    +44
═══════════════════════════════════════════════════════════════════════════
```

Results are saved to `data/eval_results_YYYYMMDD_HHMMSS.json`.

---

## Step 7 — Run the Test Suite

```bash
pip install pytest
pytest tests/ -v
```

Expected output:
```
tests/test_cases.py::TestTrainingData::test_training_file_exists         PASSED
tests/test_cases.py::TestTrainingData::test_minimum_record_count         PASSED
tests/test_cases.py::TestTrainingData::test_all_records_have_messages_key PASSED
...
tests/test_cases.py::TestModelfiles::test_expert_has_few_shot_examples   PASSED
======================== 22 passed in 0.18s =============================
```

---

## Step 8 — Extend the Training Data (Make It Yours)

To add more manufacturing conversations, append new lines to `data/manufacturing_train.jsonl`:

```json
{"messages": [
  {"role": "system", "content": "You are ManuBot..."},
  {"role": "user",   "content": "YOUR QUESTION HERE"},
  {"role": "assistant", "content": "**Root Cause:** ...\n\n**Recommended Action:** ...\n\n**Priority:** HIGH\n\n**Standard:** ISO ..."}
]}
```

Then re-create the models:
```bash
python scripts/prepare_data.py   # validate
python scripts/create_models.py  # rebuild models
```

**Best practices for training data:**
- Minimum 10 examples — more is better (aim for 50+)
- Cover all use cases your model will face
- Make assistant responses follow the exact format you want
- Vary question phrasing (don't just paraphrase the same question)
- Include edge cases and failure scenarios

---

## Step 9 (Advanced) — Level 3: LoRA Fine-Tuning

For actual weight-level training, use **Unsloth** (fast LoRA on consumer GPUs):

### Prerequisites
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+
- `pip install unsloth`

### Training script (outline)

```python
from unsloth import FastLanguageModel
import json

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.2-3b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,               # LoRA rank — higher = more capacity
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05
)

# Load our manufacturing training data
with open("data/manufacturing_train_alpaca.json") as f:
    dataset = json.load(f)

# Train...  (see Unsloth docs for full TrainingArguments)

# Export to GGUF for Ollama
model.save_pretrained_gguf("manufacturing-expert-lora", tokenizer)
```

### Import LoRA weights back to Ollama

```dockerfile
# Modelfile.lora
FROM ./manufacturing-expert-lora/manufacturing-expert-lora.Q4_K_M.gguf
SYSTEM "You are ManuBot..."
```

```bash
ollama create mfg-expert-lora -f Modelfile.lora
```

Now add `mfg-expert-lora` to the UI model selector to compare all three variants.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Cannot connect to Ollama` | Run `ollama serve` in a separate terminal |
| `Model not found` | Run `python scripts/create_models.py` |
| Model gives generic responses | Check Modelfile SYSTEM prompt is correct, rebuild model |
| UI shows blank panels | Check browser console — likely a CORS or port issue |
| Evaluation scores are 0% | Model may not be running; check `ollama list` |
| `pip install` errors | Use Python 3.9+ and a virtual environment |

---

## Key Concepts Glossary

| Term | Definition |
|------|-----------|
| **Modelfile** | Ollama's configuration file — defines base model, system prompt, parameters |
| **Few-shot** | Providing example Q&A pairs to demonstrate expected behavior |
| **Temperature** | Controls randomness (0 = deterministic, 1 = creative). Lower = more accurate for technical topics |
| **LoRA** | Low-Rank Adaptation — efficient fine-tuning that updates a small set of adapter weights |
| **GGUF** | Format for quantized model weights used by llama.cpp and Ollama |
| **RAG** | Retrieval-Augmented Generation — query a knowledge base and inject context per request |
| **OEE** | Overall Equipment Effectiveness = Availability × Performance × Quality |
| **FMEA** | Failure Mode and Effects Analysis — structured risk assessment |
| **PFMEA** | Process FMEA — applied to manufacturing processes |
| **RPN** | Risk Priority Number = Severity × Occurrence × Detection (in FMEA) |
