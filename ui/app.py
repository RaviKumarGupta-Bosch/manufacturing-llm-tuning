"""
app.py — FastAPI Backend for Manufacturing LLM Comparison UI
Serves the side-by-side model comparison interface and proxies requests to Ollama.

Start: uvicorn app:app --reload --port 8000
UI:    http://localhost:8000
"""
import json
import time
import asyncio
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Configuration ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_MODEL_BASE = "mfg-base"
DEFAULT_MODEL_EXPERT = "mfg-expert"
REQUEST_TIMEOUT = 120.0  # seconds

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Manufacturing LLM Tuning Comparison",
    description="Side-by-side comparison of base vs fine-tuned Ollama models",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Request/Response Models ────────────────────────────────────────────────────
class CompareRequest(BaseModel):
    prompt: str
    model_a: str = DEFAULT_MODEL_BASE
    model_b: str = DEFAULT_MODEL_EXPERT
    temperature: float = 0.4
    max_tokens: int = 1024


class StreamRequest(BaseModel):
    prompt: str
    model: str
    temperature: float = 0.4
    max_tokens: int = 1024


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>UI not found</h1><p>Check static/index.html</p>", status_code=404)
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/api/models")
async def list_models():
    """List all available Ollama models."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [
                {
                    "name": m["name"],
                    "size": m.get("size", 0),
                    "modified": m.get("modified_at", ""),
                    "family": m.get("details", {}).get("family", "unknown"),
                }
                for m in data.get("models", [])
            ]
            return {"models": models, "count": len(models)}
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="Cannot connect to Ollama. Ensure 'ollama serve' is running."
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Check Ollama connection health."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            has_base = any(m.startswith("mfg-base") for m in models)
            has_expert = any(m.startswith("mfg-expert") for m in models)
            return {
                "ollama": "online",
                "model_base_ready": has_base,
                "model_expert_ready": has_expert,
                "all_models": models,
                "setup_needed": not (has_base and has_expert)
            }
        except Exception:
            return {
                "ollama": "offline",
                "model_base_ready": False,
                "model_expert_ready": False,
                "setup_needed": True
            }


@app.post("/api/stream/{side}")
async def stream_single(side: str, request: StreamRequest):
    """
    Stream a response from a single model (used for parallel side-by-side streaming).
    side: 'left' or 'right' (informational only)
    """
    if side not in ("left", "right"):
        raise HTTPException(status_code=400, detail="side must be 'left' or 'right'")

    async def event_stream() -> AsyncGenerator[str, None]:
        start_time = time.time()
        total_tokens = 0
        full_response = ""

        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        yield f"data: {json.dumps({'error': f'Ollama returned {response.status_code}: {error_body.decode()}'})}\n\n"
                        return

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        token = chunk.get("response", "")
                        full_response += token
                        total_tokens += 1 if token else 0

                        event_data = {
                            "token": token,
                            "done": chunk.get("done", False),
                        }

                        if chunk.get("done"):
                            elapsed = time.time() - start_time
                            tokens_per_sec = round(total_tokens / elapsed, 1) if elapsed > 0 else 0
                            event_data.update({
                                "elapsed_ms": round(elapsed * 1000),
                                "tokens": total_tokens,
                                "tokens_per_sec": tokens_per_sec,
                                "model": request.model
                            })

                        yield f"data: {json.dumps(event_data)}\n\n"

                        if chunk.get("done"):
                            break

        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'Cannot connect to Ollama. Run: ollama serve'})}\n\n"
        except httpx.TimeoutException:
            yield f"data: {json.dumps({'error': f'Model response timed out after {REQUEST_TIMEOUT}s'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@app.get("/api/sample-prompts")
async def sample_prompts():
    """Return manufacturing sample prompts for the UI picker."""
    return {
        "prompts": [
            {
                "category": "Predictive Maintenance",
                "prompt": "CNC Machine #4 spindle vibration has increased from 0.5 mm/s to 4.2 mm/s over the last 48 hours. What should I do?"
            },
            {
                "category": "OEE Analysis",
                "prompt": "Our OEE dropped from 87% to 64% this week on Assembly Line 2. How do I diagnose and fix this?"
            },
            {
                "category": "Quality Defect",
                "prompt": "Surface porosity defects are appearing on 12% of our aluminum die castings. The defect rate was 2% last month. What changed?"
            },
            {
                "category": "Safety Emergency",
                "prompt": "A worker was exposed to a hydraulic oil mist in the press shop. What is the emergency protocol?"
            },
            {
                "category": "Equipment Fault",
                "prompt": "Robot arm on welding cell keeps triggering E-stop fault code E-47. What does this mean and how do I fix it?"
            },
            {
                "category": "FMEA",
                "prompt": "How do I apply FMEA to a new welding process we are launching next month? Give me a step-by-step approach."
            },
            {
                "category": "Lean Manufacturing",
                "prompt": "Our changeover time is 4 hours on the press line. How do I reduce it to under 30 minutes using SMED?"
            },
            {
                "category": "Heat Treatment",
                "prompt": "Temperature sensor on furnace #7 is reading 850°C but the product shows signs of incomplete heat treatment. What is wrong?"
            },
            {
                "category": "Basic Query",
                "prompt": "What is OEE?"
            },
            {
                "category": "Beginner Question",
                "prompt": "Explain predictive maintenance in simple terms."
            },
            {
                "category": "PEFT Overview",
                "prompt": "What is Parameter-Efficient Fine-Tuning (PEFT) and why is it preferred over full fine-tuning for manufacturing domain adaptation?"
            },
            {
                "category": "LoRA Technique",
                "prompt": "Explain how LoRA fine-tuning works. What are the rank, alpha, and target_modules hyperparameters and how should I choose them?"
            },
            {
                "category": "QLoRA Technique",
                "prompt": "What is QLoRA? How does 4-bit quantization combined with LoRA enable fine-tuning on a consumer GPU? What are the quality tradeoffs?"
            }
        ]
    }
