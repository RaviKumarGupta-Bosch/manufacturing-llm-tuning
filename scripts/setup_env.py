"""
setup_env.py — Environment Setup & Dependency Checker
Run this first: python scripts/setup_env.py
"""
import subprocess
import sys
import json
from pathlib import Path


def check_python_version():
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"[FAIL] Python 3.9+ required. Found: {version.major}.{version.minor}")
        return False
    print(f"[OK]   Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_ollama():
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"[OK]   Ollama: {version}")
            return True
        else:
            print("[FAIL] Ollama found but returned error")
            return False
    except FileNotFoundError:
        print("[FAIL] Ollama not found. Install from: https://ollama.com/download")
        return False
    except Exception as e:
        print(f"[FAIL] Ollama check error: {e}")
        return False


def check_ollama_running():
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            print(f"[OK]   Ollama API running. Models available: {len(models)}")
            if models:
                print(f"       Available: {', '.join(models[:5])}")
            return True, models
    except Exception:
        print("[WARN] Ollama API not responding. Start with: ollama serve")
        return False, []


def check_model(model_name: str, available_models: list):
    found = any(m.startswith(model_name.split(":")[0]) for m in available_models)
    if found:
        print(f"[OK]   Model '{model_name}' is available")
    else:
        print(f"[WARN] Model '{model_name}' not found. Pull with: ollama pull {model_name}")
    return found


def install_requirements():
    req_file = Path(__file__).parent.parent / "ui" / "requirements.txt"
    if not req_file.exists():
        print("[SKIP] requirements.txt not found")
        return
    print("\n[INFO] Installing Python dependencies...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("[OK]   Dependencies installed successfully")
    else:
        print(f"[FAIL] pip install failed:\n{result.stderr}")


def main():
    print("=" * 55)
    print("  Manufacturing LLM Tuning — Environment Check")
    print("=" * 55)

    all_ok = True

    # Python
    if not check_python_version():
        all_ok = False

    # Ollama binary
    if not check_ollama():
        all_ok = False

    # Ollama API
    api_ok, models = check_ollama_running()

    # Required models
    if api_ok:
        for model in ["llama3.2", "llama3.1"]:
            check_model(model, models)

    # Python packages
    install_requirements()

    print("\n" + "=" * 55)
    if all_ok and api_ok:
        print("[READY] Environment is set up. Next: python scripts/create_models.py")
    else:
        print("[ACTION NEEDED] Fix the items marked [FAIL] or [WARN] above.")
    print("=" * 55)


if __name__ == "__main__":
    main()
