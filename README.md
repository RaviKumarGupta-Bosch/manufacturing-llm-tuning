# Ollama Model Tuning — Manufacturing Domain
## Concept: Base Model vs Fine-Tuned Model, Side-by-Side

This project demonstrates the **full lifecycle of LLM customization** using Ollama local models, applied to a **manufacturing / industrial domain**. It covers three progressive levels of tuning, plus a live UI to compare a base model against a tuned model with the same prompt.

---

## Project Structure

```
Model-Tuning/
├── README.md                        ← This file
├── docs/
│   └── tutorial.md                  ← Full step-by-step narrative guide
├── data/
│   ├── manufacturing_train.jsonl    ← Fine-tuning training data (JSONL)
│   ├── manufacturing_test.jsonl     ← Evaluation / test prompts
│   └── knowledge_base.md           ← RAG knowledge base (manufacturing SOPs)
├── modelfiles/
│   ├── Modelfile.base               ← Minimal system prompt (baseline)
│   └── Modelfile.manufacturing      ← Domain-tuned via Modelfile
├── scripts/
│   ├── setup_env.py                 ← Checks Ollama, installs dependencies
│   ├── prepare_data.py              ← Validates and formats training data
│   ├── create_models.py             ← Creates Ollama models from Modelfiles
│   ├── evaluate_models.py           ← Runs test cases, scores responses
│   └── finetune_lora.py             ← QLoRA fine-tuning with Unsloth (Level 3)
├── ui/
│   ├── app.py                       ← FastAPI backend
│   ├── requirements.txt
│   └── static/
│       ├── index.html               ← Side-by-side comparison UI
│       ├── style.css
│       └── app.js
└── tests/
    └── test_cases.py                ← Pytest unit tests for model behavior
```

---

## The Three Levels of Ollama Model Tuning

| Level | Technique | Effort | Works With |
|-------|-----------|--------|------------|
| **1** | **Modelfile** — system prompt + few-shot examples | Minutes | Any Ollama model |
| **2** | **RAG** — retrieval-augmented generation | Hours | Any Ollama model |
| **3** | **LoRA Fine-tuning** — gradient-based weight update | Days | Models supporting GGUF export |

This project implements all three and lets you compare them live.

---

## Quick Start (5 minutes)

```bash
# 1. Clone this repo
git clone https://github.com/RaviKumarGupta-Bosch/manufacturing-llm-tuning
cd manufacturing-llm-tuning

# 2. Install Python dependencies
pip install -r ui/requirements.txt

# 3. Pull base model (if not already present)
ollama pull llama2

# 4. Create the manufacturing-tuned Modelfile model
python scripts/create_models.py

# 5. Start the comparison UI
cd ui
uvicorn app:app --reload --port 8000

# 6. Open browser
# http://localhost:8000
```

---

## Use Cases Covered (Manufacturing Domain)

1. **Predictive Maintenance** — "When should I replace the bearing on Line 3?"
2. **Defect Root Cause Analysis** — "Surface cracks appearing on output. What is the cause?"
3. **Equipment Fault Diagnosis** — "CNC spindle vibration above threshold. Steps?"
4. **Safety Compliance** — "Operator exposed to chemical X. Protocol?"
5. **OEE / KPI Analysis** — "OEE dropped from 85% to 67%. Investigate."
6. **Production Scheduling** — "Optimize shift plan given current downtime."
7. **PEFT / LoRA / QLoRA** — "Explain how LoRA fine-tuning works for domain adaptation."

---

## What "Training" Means Here

```
┌─────────────────────────────────────────────────────────┐
│  BASE MODEL (llama2)                                    │
│  General knowledge, no manufacturing context            │
│  System prompt: "You are a helpful assistant."          │
└─────────────────────────────────────────────────────────┘
           │
           ▼  Fine-Tuning / Modelfile Customization
┌─────────────────────────────────────────────────────────┐
│  TUNED MODEL (mfg-expert)                               │
│  Manufacturing domain knowledge baked in                │
│  Knows OEE, FMEA, ISO standards, SPC, LEAN              │
│  Responds in structured maintenance/quality format      │
└─────────────────────────────────────────────────────────┘
```

See [docs/tutorial.md](docs/tutorial.md) for the complete walkthrough.

---

## Level 3: QLoRA Fine-Tuning

For actual weight-level training on consumer GPUs:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install bitsandbytes datasets trl

python scripts/prepare_data.py    # export to Alpaca format
python scripts/finetune_lora.py   # run QLoRA (~1-3h on RTX 4090)
ollama create mfg-expert-qlora -f modelfiles/Modelfile.qlora
```

See [scripts/finetune_lora.py](scripts/finetune_lora.py) for the full annotated training script.

---

## License

MIT
