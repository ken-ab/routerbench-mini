from __future__ import annotations

from routerbench_mini.normalization import (
    canonical_answer,
    extract_choice,
    extract_number,
    normalize_tool_call,
    parse_json_object,
)


def test_extract_number_uses_last_number_and_normalizes_integer_float() -> None:
    assert extract_number("The subtotal was 1,200, but the answer is 42.0.") == "42"


def test_canonical_math_answer_handles_gsm8k_style_text() -> None:
    assert canonical_answer("math", "Add the values. #### 1,234") == "1234"


def test_extract_choice_supports_five_option_questions() -> None:
    assert extract_choice("E") == "E"


def test_parse_json_object_from_fenced_text() -> None:
    text = """Here is the tool call:

```json
{"name": "get_weather", "arguments": {"city": "Paris"}}
```
"""

    assert parse_json_object(text) == {"name": "get_weather", "arguments": {"city": "Paris"}}


def test_normalize_tool_call_accepts_nested_function_shape() -> None:
    raw = {"function": {"name": "search_papers", "arguments": {"query": "LLM routing"}}}

    assert normalize_tool_call(raw) == {
        "name": "search_papers",
        "arguments": {"query": "LLM routing"},
    }
