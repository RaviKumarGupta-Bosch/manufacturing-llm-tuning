"""
evaluate_models.py — Automated Model Evaluation
Runs the test prompts against both models and scores the responses.

Run: python scripts/evaluate_models.py
"""
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime


TEST_FILE = Path(__file__).parent.parent / "data" / "manufacturing_test.jsonl"
RESULTS_DIR = Path(__file__).parent.parent / "data"
OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS_TO_EVALUATE = [
    {"name": "mfg-base",   "label": "Base (Untrained)"},
    {"name": "mfg-expert", "label": "Manufacturing Expert (Tuned)"},
]


def load_test_cases() -> list[dict]:
    cases = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def query_model(model_name: str, prompt: str, timeout: int = 60) -> tuple[str, float]:
    """Query Ollama model and return (response, elapsed_seconds)."""
    payload = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512
        }
    }).encode()

    start = time.time()
    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            elapsed = time.time() - start
            return data.get("response", ""), elapsed
    except urllib.error.URLError as e:
        return f"[ERROR: {e}]", 0.0
    except Exception as e:
        return f"[ERROR: {e}]", 0.0


def score_response(response: str, expected_keywords: list[str], expected_format: list[str]) -> dict:
    """Score a model response against expected keywords and format markers."""
    response_lower = response.lower()

    # Keyword coverage score (0–100)
    found_keywords = [kw for kw in expected_keywords if kw.lower() in response_lower]
    keyword_score = round(len(found_keywords) / len(expected_keywords) * 100) if expected_keywords else 0

    # Format compliance score (0–100)
    found_format = [f for f in expected_format if f.lower() in response_lower]
    format_score = round(len(found_format) / len(expected_format) * 100) if expected_format else 0

    # Length score — penalize too short or too long
    word_count = len(response.split())
    if word_count < 50:
        length_score = 30
    elif 50 <= word_count <= 600:
        length_score = 100
    else:
        length_score = 80  # slightly penalize very long responses

    overall = round((keyword_score * 0.4 + format_score * 0.35 + length_score * 0.25))

    return {
        "keyword_score": keyword_score,
        "format_score": format_score,
        "length_score": length_score,
        "overall": overall,
        "found_keywords": found_keywords,
        "word_count": word_count
    }


def print_comparison_table(results: list[dict]):
    """Print a formatted comparison table."""
    print(f"\n{'='*75}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*75}")
    print(f"  {'Test ID':<8} {'Category':<28} {'Base Score':>10} {'Expert Score':>13} {'Delta':>6}")
    print(f"  {'-'*7} {'-'*27} {'-'*10} {'-'*13} {'-'*6}")

    total_base = 0
    total_expert = 0

    for r in results:
        base = r.get("mfg-base", {}).get("score", {}).get("overall", 0)
        expert = r.get("mfg-expert", {}).get("score", {}).get("overall", 0)
        delta = expert - base
        delta_str = f"+{delta}" if delta >= 0 else str(delta)
        total_base += base
        total_expert += expert
        print(f"  {r['id']:<8} {r['category']:<28} {base:>9}% {expert:>12}% {delta_str:>6}")

    n = len(results)
    if n > 0:
        avg_base = round(total_base / n)
        avg_expert = round(total_expert / n)
        avg_delta = avg_expert - avg_base
        print(f"  {'-'*7} {'-'*27} {'-'*10} {'-'*13} {'-'*6}")
        delta_str = f"+{avg_delta}" if avg_delta >= 0 else str(avg_delta)
        print(f"  {'AVERAGE':<8} {'':<28} {avg_base:>9}% {avg_expert:>12}% {delta_str:>6}")

    print(f"{'='*75}")


def main():
    print("=" * 55)
    print("  Manufacturing LLM — Model Evaluation")
    print("=" * 55)

    test_cases = load_test_cases()
    print(f"\nLoaded {len(test_cases)} test cases from {TEST_FILE.name}")

    all_results = []

    for i, case in enumerate(test_cases, 1):
        test_id = case.get("id", f"T{i:03d}")
        category = case.get("category", "unknown")
        prompt = case.get("prompt", "")
        expected_keywords = case.get("expected_keywords", [])
        expected_format = case.get("expected_format", [])

        print(f"\n[{i}/{len(test_cases)}] {test_id} — {category}")
        print(f"  Prompt: {prompt[:80]}...")

        result = {"id": test_id, "category": category, "prompt": prompt}

        for model_cfg in MODELS_TO_EVALUATE:
            model_name = model_cfg["name"]
            label = model_cfg["label"]
            print(f"  Querying {label}...", end=" ", flush=True)
            response, elapsed = query_model(model_name, prompt)
            score = score_response(response, expected_keywords, expected_format)
            print(f"Score: {score['overall']}% ({elapsed:.1f}s)")
            result[model_name] = {
                "response": response,
                "elapsed": round(elapsed, 2),
                "score": score
            }

        all_results.append(result)

    # Print comparison table
    print_comparison_table(all_results)

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"eval_results_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] Results saved to: {output_file}")

    # Summary
    base_scores = [r.get("mfg-base", {}).get("score", {}).get("overall", 0) for r in all_results]
    expert_scores = [r.get("mfg-expert", {}).get("score", {}).get("overall", 0) for r in all_results]
    if base_scores and expert_scores:
        improvement = round(sum(expert_scores) / len(expert_scores) - sum(base_scores) / len(base_scores))
        print(f"\n[SUMMARY] Tuning improvement: +{improvement} average score points")
        print(f"  Base model avg   : {round(sum(base_scores)/len(base_scores))}%")
        print(f"  Expert model avg : {round(sum(expert_scores)/len(expert_scores))}%")


if __name__ == "__main__":
    main()
