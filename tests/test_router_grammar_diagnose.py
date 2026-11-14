"""``diagnose`` explains a constrained output that is not schema-valid. It never makes one valid."""

from __future__ import annotations

from inferference.benchkit.runners.router_grammar import diagnose, validate

VALID = '{"tool": "calculator", "confidence": "high"}'


def test_a_valid_output_is_valid_and_the_validator_is_unchanged() -> None:
    assert diagnose(VALID, "stop", range(12), 48) == "valid"
    assert validate(VALID)["schema_ok"] is True
    assert validate('{"tool": "calculator"')["schema_ok"] is False  # a cut JSON stays invalid


def test_a_whitespace_run_cut_at_the_token_budget_is_named() -> None:
    text = "{\n" + "\n" * 60 + '  "tool"'
    why = diagnose(text, "length", range(48), 48)
    assert why.startswith("cut at max_tokens") and "mostly whitespace" in why and "48/48" in why
    # The hint matches the grammar: compact JSON by default, a bounded opt-in for whitespace.
    assert "compact default allows none" in why and "allow_whitespace" in why
    assert "bounds no whitespace run" not in why


def test_the_docstring_describes_the_compact_default_and_the_bounded_option() -> None:
    doc = diagnose.__doc__ or ""
    assert "compact JSON" in doc and "allow_whitespace=True" in doc and "MAX_WHITESPACE_CNT" in doc
    assert "any_whitespace" not in doc and "no ``max_whitespace_cnt``" not in doc


def test_a_cut_output_with_little_whitespace_is_named_without_the_whitespace_hint() -> None:
    why = diagnose('{"tool": "calcul', "length", range(48), 48)
    assert why.startswith("cut at max_tokens") and "mostly whitespace" not in why


def test_finished_but_wrong_outputs_are_told_apart() -> None:
    assert "empty output" in diagnose("  ", "stop", range(3), 48)
    assert "not JSON" in diagnose("tool: calculator", "stop", range(5), 48)
    assert "fails the schema" in diagnose('{"tool": "toaster", "confidence": "high"}', "stop", range(9), 48)
