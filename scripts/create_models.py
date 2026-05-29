"""
create_models.py — Create Ollama Models from Modelfiles
This script reads Modelfile.base and Modelfile.manufacturing
and registers them as Ollama models for side-by-side comparison.

Run: python scripts/create_models.py
"""
import subprocess
import sys
import time
from pathlib import Path


MODELFILES_DIR = Path(__file__).parent.parent / "modelfiles"

MODELS = [
    {
        "name": "mfg-base",
        "modelfile": MODELFILES_DIR / "Modelfile.base",
        "description": "Base model — minimal system prompt (before training)",
        "display": "Base Model (Untrained)"
    },
    {
        "name": "mfg-expert",
        "modelfile": MODELFILES_DIR / "Modelfile.manufacturing",
        "description": "Manufacturing domain expert — tuned via Modelfile (after training)",
        "display": "Manufacturing Expert (Tuned)"
    }
]


def ollama_is_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


def create_model(name: str, modelfile_path: Path) -> bool:
    """Create an Ollama model from a Modelfile."""
    if not modelfile_path.exists():
        print(f"  [FAIL] Modelfile not found: {modelfile_path}")
        return False

    print(f"  Creating '{name}' from {modelfile_path.name}...")
    result = subprocess.run(
        ["ollama", "create", name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode == 0:
        print(f"  [OK]   Model '{name}' created successfully")
        return True
    else:
        # Ollama often prints progress to stderr even on success
        if "success" in result.stderr.lower() or "success" in result.stdout.lower():
            print(f"  [OK]   Model '{name}' created")
            return True
        print(f"  [FAIL] Failed to create '{name}'")
        print(f"         stdout: {result.stdout.strip()}")
        print(f"         stderr: {result.stderr.strip()}")
        return False


def list_created_models():
    """Show all models currently in Ollama."""
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        print("\n  Current Ollama models:")
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")


def test_model(name: str, test_prompt: str = "What is OEE?") -> bool:
    """Quick smoke-test of a created model."""
    import urllib.request
    import json

    payload = json.dumps({
        "model": name,
        "prompt": test_prompt,
        "stream": False,
        "options": {"num_predict": 60}
    }).encode()

    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            response = data.get("response", "").strip()
            preview = response[:120] + "..." if len(response) > 120 else response
            print(f"  [TEST] '{name}' response preview:")
            print(f"         \"{preview}\"")
            return bool(response)
    except Exception as e:
        print(f"  [WARN] Smoke test failed for '{name}': {e}")
        return False


def main():
    print("=" * 55)
    print("  Manufacturing LLM — Model Creation")
    print("=" * 55)

    if not ollama_is_running():
        print("\n[FAIL] Ollama API is not running.")
        print("  Start it with: ollama serve")
        print("  Then re-run this script.")
        sys.exit(1)

    print(f"\nCreating {len(MODELS)} model(s)...\n")
    created = []

    for model_config in MODELS:
        print(f"[{model_config['display']}]")
        print(f"  {model_config['description']}")
        success = create_model(model_config["name"], model_config["modelfile"])
        if success:
            created.append(model_config["name"])
        print()

    # Brief pause to let Ollama settle
    if created:
        time.sleep(2)

    # Smoke test each created model
    print("\n--- Smoke Tests ---")
    for model_config in MODELS:
        if model_config["name"] in created:
            test_model(model_config["name"])
            print()

    list_created_models()

    print("\n" + "=" * 55)
    if len(created) == len(MODELS):
        print(f"[READY] All {len(MODELS)} models created.")
        print("\n  Model names for the UI:")
        for m in MODELS:
            print(f"    {m['name']:<20} — {m['display']}")
        print("\n  Next: cd ui && uvicorn app:app --reload --port 8000")
        print("  Then open: http://localhost:8000")
    else:
        failed = [m["name"] for m in MODELS if m["name"] not in created]
        print(f"[WARN] {len(failed)} model(s) failed to create: {', '.join(failed)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
