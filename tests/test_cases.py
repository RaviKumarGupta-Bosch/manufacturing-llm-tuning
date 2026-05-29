"""
tests/test_cases.py — Pytest unit tests for model behavior validation.
Tests both the data pipeline and model response quality.

Run: pytest tests/ -v
"""
import json
import pytest
from pathlib import Path
import sys

# Allow imports from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

DATA_DIR = Path(__file__).parent.parent / "data"


# ══════════════════════════════════════════════════════════════════════════════
# Data Pipeline Tests
# ══════════════════════════════════════════════════════════════════════════════
class TestTrainingData:
    """Validate the training JSONL structure and content."""

    def setup_method(self):
        self.records = []
        train_file = DATA_DIR / "manufacturing_train.jsonl"
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def test_training_file_exists(self):
        assert (DATA_DIR / "manufacturing_train.jsonl").exists(), \
            "manufacturing_train.jsonl not found in data/"

    def test_minimum_record_count(self):
        assert len(self.records) >= 5, \
            f"Expected >= 5 training records, found {len(self.records)}"

    def test_all_records_have_messages_key(self):
        for i, r in enumerate(self.records):
            assert "messages" in r, f"Record {i+1} missing 'messages' key"

    def test_all_records_have_system_message(self):
        for i, r in enumerate(self.records):
            roles = [m["role"] for m in r["messages"]]
            assert "system" in roles, f"Record {i+1} has no system message"
            assert roles[0] == "system", f"Record {i+1}: system must be first message"

    def test_all_records_have_user_and_assistant(self):
        for i, r in enumerate(self.records):
            roles = [m["role"] for m in r["messages"]]
            assert "user" in roles, f"Record {i+1} missing user message"
            assert "assistant" in roles, f"Record {i+1} missing assistant message"

    def test_no_empty_content(self):
        for i, r in enumerate(self.records):
            for j, m in enumerate(r["messages"]):
                assert m.get("content", "").strip(), \
                    f"Record {i+1}, message {j+1} has empty content"

    def test_system_prompt_mentions_manufacturing(self):
        keywords = ["manufacturing", "manubot", "oee", "fmea", "maintenance"]
        for i, r in enumerate(self.records):
            system_content = next(
                (m["content"].lower() for m in r["messages"] if m["role"] == "system"),
                ""
            )
            has_keyword = any(k in system_content for k in keywords)
            assert has_keyword, \
                f"Record {i+1}: system prompt does not mention manufacturing context"

    def test_assistant_responses_are_substantial(self):
        """Each assistant response should be at least 100 words."""
        for i, r in enumerate(self.records):
            assistant_content = next(
                (m["content"] for m in r["messages"] if m["role"] == "assistant"),
                ""
            )
            word_count = len(assistant_content.split())
            assert word_count >= 100, \
                f"Record {i+1}: assistant response only {word_count} words (expected >= 100)"

    def test_assistant_responses_have_structure(self):
        """Expert responses should contain structured headers."""
        structure_markers = ["**", "root cause", "action", "priority", "standard"]
        for i, r in enumerate(self.records):
            content = next(
                (m["content"].lower() for m in r["messages"] if m["role"] == "assistant"),
                ""
            )
            found = sum(1 for m in structure_markers if m in content)
            assert found >= 2, \
                f"Record {i+1}: response lacks structured format (found {found}/5 markers)"


class TestTestData:
    """Validate the test JSONL structure."""

    def setup_method(self):
        self.records = []
        test_file = DATA_DIR / "manufacturing_test.jsonl"
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def test_test_file_exists(self):
        assert (DATA_DIR / "manufacturing_test.jsonl").exists()

    def test_minimum_test_count(self):
        assert len(self.records) >= 5

    def test_all_required_fields_present(self):
        required = ["id", "category", "prompt", "expected_keywords"]
        for i, r in enumerate(self.records):
            for field in required:
                assert field in r, f"Test record {i+1} missing field '{field}'"

    def test_unique_test_ids(self):
        ids = [r["id"] for r in self.records]
        assert len(ids) == len(set(ids)), "Duplicate test IDs found"

    def test_prompts_are_non_empty(self):
        for i, r in enumerate(self.records):
            assert r["prompt"].strip(), f"Test record {i+1} has empty prompt"

    def test_expected_keywords_are_lists(self):
        for i, r in enumerate(self.records):
            assert isinstance(r["expected_keywords"], list), \
                f"Test record {i+1}: expected_keywords must be a list"
            assert len(r["expected_keywords"]) >= 2, \
                f"Test record {i+1}: need at least 2 expected_keywords"

    def test_categories_are_valid(self):
        valid_categories = {
            "predictive_maintenance", "quality_defect", "oee_analysis",
            "safety", "fmea", "lean", "equipment_fault", "quality_system"
        }
        for i, r in enumerate(self.records):
            cat = r.get("category", "")
            assert cat in valid_categories, \
                f"Test record {i+1}: unknown category '{cat}'. " \
                f"Valid: {valid_categories}"


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation Logic Tests
# ══════════════════════════════════════════════════════════════════════════════
class TestScoringLogic:
    """Unit tests for the evaluate_models.py scoring function."""

    def _score(self, response, keywords, fmt_markers):
        """Inline scoring logic (mirrors evaluate_models.py)."""
        response_lower = response.lower()
        found_kw = [k for k in keywords if k.lower() in response_lower]
        kw_score = round(len(found_kw) / len(keywords) * 100) if keywords else 0

        found_fmt = [f for f in fmt_markers if f.lower() in response_lower]
        fmt_score = round(len(found_fmt) / len(fmt_markers) * 100) if fmt_markers else 0

        wc = len(response.split())
        if wc < 50:
            length_score = 30
        elif 50 <= wc <= 600:
            length_score = 100
        else:
            length_score = 80

        overall = round(kw_score * 0.4 + fmt_score * 0.35 + length_score * 0.25)
        return {"keyword_score": kw_score, "format_score": fmt_score,
                "length_score": length_score, "overall": overall}

    def test_perfect_response_scores_100(self):
        response = (
            "Root Cause: bearing wear. Recommended Action: replace bearing. "
            "Priority Level: HIGH. Relevant Standard: ISO 10816. "
            "bearing vibration maintenance ISO " * 20
        )
        result = self._score(
            response,
            ["bearing", "vibration", "maintenance", "ISO"],
            ["Root Cause", "Priority Level"]
        )
        assert result["keyword_score"] == 100
        assert result["format_score"] == 100
        assert result["length_score"] == 100

    def test_empty_response_penalised(self):
        result = self._score("", ["bearing"], ["Root Cause"])
        assert result["keyword_score"] == 0
        assert result["length_score"] == 30

    def test_partial_keyword_match(self):
        response = "The bearing should be checked. " * 20
        result = self._score(response, ["bearing", "OEE", "FMEA", "ISO"], [])
        assert result["keyword_score"] == 25  # 1 of 4

    def test_short_response_gets_low_length_score(self):
        result = self._score("Short answer.", ["anything"], [])
        assert result["length_score"] == 30

    def test_scores_are_bounded_0_to_100(self):
        for response in ["", "word " * 10, "word " * 300]:
            result = self._score(response, ["word"], ["word"])
            for key in ("keyword_score", "format_score", "length_score", "overall"):
                assert 0 <= result[key] <= 100, f"{key} out of range for response length {len(response)}"


# ══════════════════════════════════════════════════════════════════════════════
# Modelfile Tests
# ══════════════════════════════════════════════════════════════════════════════
class TestModelfiles:
    """Validate the Modelfile contents."""

    MODELFILE_DIR = Path(__file__).parent.parent / "modelfiles"

    def _read(self, filename):
        return (self.MODELFILE_DIR / filename).read_text(encoding="utf-8")

    def test_base_modelfile_exists(self):
        assert (self.MODELFILE_DIR / "Modelfile.base").exists()

    def test_expert_modelfile_exists(self):
        assert (self.MODELFILE_DIR / "Modelfile.manufacturing").exists()

    def test_base_modelfile_has_from_directive(self):
        content = self._read("Modelfile.base")
        assert content.strip().startswith("FROM"), \
            "Modelfile.base must start with FROM directive"

    def test_expert_modelfile_has_from_directive(self):
        content = self._read("Modelfile.manufacturing")
        assert content.strip().startswith("FROM"), \
            "Modelfile.manufacturing must start with FROM directive"

    def test_expert_has_system_prompt(self):
        content = self._read("Modelfile.manufacturing")
        assert "SYSTEM" in content, "Expert Modelfile must contain SYSTEM prompt"
        assert len(content) > 500, "Expert Modelfile system prompt seems too short"

    def test_expert_has_lower_temperature_than_base(self):
        base = self._read("Modelfile.base")
        expert = self._read("Modelfile.manufacturing")

        def extract_temp(text):
            for line in text.split("\n"):
                if "temperature" in line.lower() and "PARAMETER" in line:
                    parts = line.split()
                    try:
                        return float(parts[-1])
                    except ValueError:
                        pass
            return None

        base_temp = extract_temp(base)
        expert_temp = extract_temp(expert)

        if base_temp is not None and expert_temp is not None:
            assert expert_temp < base_temp, \
                f"Expert model temperature ({expert_temp}) should be lower than base ({base_temp})"

    def test_expert_modelfile_mentions_manufacturing_keywords(self):
        content = self._read("Modelfile.manufacturing").lower()
        required_keywords = ["oee", "fmea", "maintenance", "iso"]
        for kw in required_keywords:
            assert kw in content, \
                f"Expert Modelfile does not mention '{kw}' — domain context may be incomplete"

    def test_expert_has_few_shot_examples(self):
        content = self._read("Modelfile.manufacturing")
        message_count = content.count("MESSAGE user")
        assert message_count >= 1, \
            "Expert Modelfile should have at least 1 few-shot MESSAGE example"
