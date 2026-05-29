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
│   └── evaluate_models.py          ← Runs test cases, scores responses
├── ui/
│   ├── app.py                       ← FastAPI backend
│   ├── requirements.txt
│   └── static/
│       ├── index.html               ← Side-by-side comparison UI
│       ├── style.css
│       └── app.js
└── tests/
    ├── test_cases.py                ← Pytest unit tests for model behavior
    └── test_data.py                 ← Data validation tests
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
# 1. Clone / open this folder in terminal
cd "c:\Users\PLT3KOR\Documents\AI\Public\Model-Tuning"

# 2. Install Python dependencies
pip install -r ui/requirements.txt

# 3. Pull base model (if not already present)
ollama pull llama3.2

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

---

## What "Training" Means Here

```
┌─────────────────────────────────────────────────────────┐
│  BASE MODEL (llama3.2)                                  │
│  General knowledge, no manufacturing context            │
│  System prompt: "You are a helpful assistant."          │
└─────────────────────────────────────────────────────────┘
           │
           ▼  Fine-Tuning / Modelfile Customization
┌─────────────────────────────────────────────────────────┐
│  TUNED MODEL (manufacturing-expert)                     │
│  Manufacturing domain knowledge baked in                │
│  Knows OEE, FMEA, ISO standards, SPC, LEAN             │
│  Responds in structured maintenance/quality format      │
└─────────────────────────────────────────────────────────┘
```

See [docs/tutorial.md](docs/tutorial.md) for the complete walkthrough.
