"""
prepare_data.py — Training Data Validation and Statistics
Validates the JSONL training data format and prints summary stats.
Run: python scripts/prepare_data.py
"""
import json
import sys
from pathlib import Path
from collections import Counter


DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_FILE = DATA_DIR / "manufacturing_train.jsonl"
TEST_FILE = DATA_DIR / "manufacturing_test.jsonl"


def load_jsonl(filepath: Path) -> list[dict]:
    """Load a JSONL file and return list of records."""
    records = []
    errors = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append(f"  Line {line_no}: {e}")
    if errors:
        print(f"[WARN] JSON errors in {filepath.name}:")
        for err in errors:
            print(err)
    return records


def validate_training_record(record: dict, index: int) -> list[str]:
    """Validate a single training record. Returns list of error strings."""
    issues = []
    if "messages" not in record:
        issues.append(f"  Record {index}: missing 'messages' key")
        return issues

    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        issues.append(f"  Record {index}: 'messages' must be a list with >= 2 entries")
        return issues

    roles = [m.get("role") for m in messages]
    if roles[0] != "system":
        issues.append(f"  Record {index}: first message should be role='system'")

    has_user = "user" in roles
    has_assistant = "assistant" in roles
    if not has_user:
        issues.append(f"  Record {index}: no 'user' message found")
    if not has_assistant:
        issues.append(f"  Record {index}: no 'assistant' message found")

    for msg in messages:
        if "content" not in msg or not msg["content"].strip():
            issues.append(f"  Record {index}: message has empty 'content'")

    return issues


def validate_test_record(record: dict, index: int) -> list[str]:
    """Validate a single test record."""
    issues = []
    required = ["id", "category", "prompt", "expected_keywords"]
    for key in required:
        if key not in record:
            issues.append(f"  Record {index}: missing key '{key}'")
    if "expected_keywords" in record and len(record["expected_keywords"]) < 2:
        issues.append(f"  Record {index}: at least 2 expected_keywords required")
    return issues


def print_training_stats(records: list[dict]):
    """Print statistics about training data."""
    total_tokens_approx = 0
    assistant_lengths = []
    user_lengths = []

    for record in records:
        for msg in record.get("messages", []):
            words = len(msg.get("content", "").split())
            approx_tokens = int(words * 1.35)  # rough token estimate
            total_tokens_approx += approx_tokens
            if msg["role"] == "assistant":
                assistant_lengths.append(words)
            elif msg["role"] == "user":
                user_lengths.append(words)

    print(f"\n  Training Records      : {len(records)}")
    print(f"  Approx Total Tokens   : {total_tokens_approx:,}")
    if assistant_lengths:
        avg_resp = sum(assistant_lengths) // len(assistant_lengths)
        print(f"  Avg Response Length   : {avg_resp} words")
    if user_lengths:
        avg_prompt = sum(user_lengths) // len(user_lengths)
        print(f"  Avg Prompt Length     : {avg_prompt} words")


def print_test_stats(records: list[dict]):
    """Print statistics about test data."""
    categories = Counter(r.get("category", "unknown") for r in records)
    print(f"\n  Test Records          : {len(records)}")
    print("  By Category:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat:<30} {count}")


def export_for_lora(records: list[dict], output_path: Path):
    """Export data in Alpaca format for LoRA fine-tuning tools (unsloth/axolotl)."""
    alpaca_records = []
    for record in records:
        messages = record.get("messages", [])
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        if user_msg and assistant_msg:
            alpaca_records.append({
                "instruction": user_msg,
                "input": "",
                "output": assistant_msg,
                "system": system_msg
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_records, f, indent=2, ensure_ascii=False)
    print(f"\n  [OK] LoRA/Alpaca format exported to: {output_path}")


def main():
    print("=" * 55)
    print("  Manufacturing Training Data Validator")
    print("=" * 55)

    # Validate training data
    print(f"\n[1] Validating Training Data: {TRAIN_FILE.name}")
    train_records = load_jsonl(TRAIN_FILE)
    all_issues = []
    for i, record in enumerate(train_records, 1):
        all_issues.extend(validate_training_record(record, i))

    if all_issues:
        print(f"[WARN] {len(all_issues)} validation issue(s):")
        for issue in all_issues:
            print(issue)
    else:
        print(f"[OK]   All {len(train_records)} training records are valid")

    print_training_stats(train_records)

    # Validate test data
    print(f"\n[2] Validating Test Data: {TEST_FILE.name}")
    test_records = load_jsonl(TEST_FILE)
    test_issues = []
    for i, record in enumerate(test_records, 1):
        test_issues.extend(validate_test_record(record, i))

    if test_issues:
        print(f"[WARN] {len(test_issues)} test validation issue(s):")
        for issue in test_issues:
            print(issue)
    else:
        print(f"[OK]   All {len(test_records)} test records are valid")

    print_test_stats(test_records)

    # Export Alpaca format for LoRA tools
    alpaca_output = DATA_DIR / "manufacturing_train_alpaca.json"
    print(f"\n[3] Exporting LoRA/Alpaca format...")
    export_for_lora(train_records, alpaca_output)

    print("\n" + "=" * 55)
    print("[DONE] Data preparation complete.")
    print("  Training data: data/manufacturing_train.jsonl")
    print("  LoRA format:   data/manufacturing_train_alpaca.json")
    print("  Test data:     data/manufacturing_test.jsonl")
    print("=" * 55)


if __name__ == "__main__":
    main()
