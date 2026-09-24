"""Tests for projects/03_bfcl_tool_calling.

The project folder is named with a numeric prefix (03_bfcl_tool_calling), which is
not a valid Python identifier, so its modules cannot be reached with a normal
`import projects.03_bfcl_tool_calling.foo` statement. We add the project directory
to sys.path directly (matching the convention used by tests/test_02_hotpotqa.py)
so its modules import as plain top-level names.

Pure-logic tests (schema conversion, AST-equivalent scoring) run always, with no
network or Ollama required. Tests that need a live Ollama server are marked
`@pytest.mark.live` (deselect with `-m "not live"`), matching the project-wide
convention in pyproject.toml. One test needs a live download of the real BFCL
data from GitHub; it's marked `@pytest.mark.live` too and skips gracefully if the
network isn't reachable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent / "projects" / "03_bfcl_tool_calling"
sys.path.insert(0, str(PROJECT_DIR))

from schema_convert import bfcl_function_to_openai_tool, build_tools  # noqa: E402
from scoring import match_call, score_case, values_equal  # noqa: E402

# ---------------------------------------------------------------------------
# schema_convert -- pure logic, no network
# ---------------------------------------------------------------------------


def test_dict_type_becomes_object_with_nested_properties():
    fn = {
        "name": "calculate_triangle_area",
        "description": "Area of a triangle.",
        "parameters": {
            "type": "dict",
            "properties": {
                "base": {"type": "integer", "description": "base"},
                "height": {"type": "integer", "description": "height"},
            },
            "required": ["base", "height"],
        },
    }
    tool = bfcl_function_to_openai_tool(fn)
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "calculate_triangle_area"
    params = tool["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["base"] == {"type": "integer", "description": "base"}
    assert params["required"] == ["base", "height"]


def test_float_and_tuple_and_array_types_map_correctly():
    fn = {
        "name": "f",
        "parameters": {
            "type": "dict",
            "properties": {
                "radius": {"type": "float"},
                "coords": {"type": "tuple", "items": {"type": "float"}},
                "tags": {"type": "array", "items": {"type": "string"}},
                "anything": {"type": "any"},
            },
            "required": ["radius"],
        },
    }
    params = bfcl_function_to_openai_tool(fn)["function"]["parameters"]
    assert params["properties"]["radius"] == {"type": "number"}
    assert params["properties"]["coords"] == {"type": "array", "items": {"type": "number"}}
    assert params["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
    # "any" has no type constraint at all
    assert "type" not in params["properties"]["anything"]


def test_build_tools_handles_multiple_functions_in_one_case():
    case = {
        "function": [
            {"name": "a", "parameters": {"type": "dict", "properties": {}}},
            {"name": "b", "parameters": {"type": "dict", "properties": {}}},
        ]
    }
    tools = build_tools(case)
    assert [t["function"]["name"] for t in tools] == ["a", "b"]


# ---------------------------------------------------------------------------
# scoring -- pure logic, no network
# ---------------------------------------------------------------------------


def test_values_equal_numeric_string_coercion():
    # llama3.2:3b in practice returns numeric args as strings, e.g. "10" for 10.
    assert values_equal("10", 10)
    assert values_equal(10, "10")
    assert values_equal(5.0, 5)


def test_values_equal_case_insensitive_strings():
    assert values_equal("Copper", "copper")
    assert not values_equal("copper", "aluminum")


def test_values_equal_bool_from_string():
    assert values_equal("true", True)
    assert values_equal(True, True)
    assert not values_equal("false", True)


def test_match_call_optional_param_may_be_omitted():
    gt = {"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}
    ok, reason = match_call("calculate_triangle_area", {"base": 10, "height": 5}, gt)
    assert ok, reason


def test_match_call_wrong_function_name_fails():
    gt = {"foo": {"x": [1]}}
    ok, reason = match_call("bar", {"x": 1}, gt)
    assert not ok
    assert "name mismatch" in reason


def test_match_call_hallucinated_extra_arg_fails():
    gt = {"foo": {"x": [1]}}
    ok, reason = match_call("foo", {"x": 1, "y": 2}, gt)
    assert not ok
    assert "hallucinated" in reason


def test_match_call_missing_required_arg_fails():
    gt = {"foo": {"x": [1]}}
    ok, reason = match_call("foo", {}, gt)
    assert not ok


def test_score_case_single_call_correct():
    predicted = [("math.factorial", {"number": 5})]
    gt = [{"math.factorial": {"number": [5]}}]
    result = score_case(predicted, gt)
    assert result["correct"]


def test_score_case_call_count_mismatch():
    predicted = [("f", {"x": 1})]
    gt = [{"f": {"x": [1]}}, {"g": {"y": [2]}}]
    result = score_case(predicted, gt)
    assert not result["correct"]
    assert "count mismatch" in result["reason"]


def test_score_case_parallel_calls_matched_out_of_order():
    # Model emits the two calls in the opposite order to ground truth -- BFCL
    # scores parallel calls as an unordered set, so this must still be correct.
    predicted = [
        ("spotify.play", {"artist": "Maroon 5", "duration": 15}),
        ("spotify.play", {"artist": "Taylor Swift", "duration": 20}),
    ]
    gt = [
        {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
        {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}},
    ]
    result = score_case(predicted, gt)
    assert result["correct"]


def test_score_case_parallel_calls_no_valid_assignment():
    predicted = [
        ("spotify.play", {"artist": "Maroon 5", "duration": 999}),
        ("spotify.play", {"artist": "Taylor Swift", "duration": 20}),
    ]
    gt = [
        {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
        {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}},
    ]
    result = score_case(predicted, gt)
    assert not result["correct"]


# ---------------------------------------------------------------------------
# Live tests -- need network (real BFCL data) and/or a running Ollama server
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_real_bfcl_data_loads_and_shapes_are_sane():
    from bfcl_data import load_category

    try:
        cases = load_category("simple_python")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not fetch real BFCL data: {exc}")
    assert len(cases) > 50
    case = cases[0]
    assert case["question"] and case["question"][0]["role"] == "user"
    assert case["function"]
    assert case["ground_truth"]


@pytest.mark.live
def test_live_ollama_end_to_end_on_one_case():
    """Full pipeline against a real local Ollama model on one real BFCL case."""
    import httpx

    try:
        httpx.get("http://127.0.0.1:11434/api/tags", timeout=3).raise_for_status()
    except Exception:  # noqa: BLE001
        pytest.skip("no local Ollama server reachable at 127.0.0.1:11434")

    # Two separate things have to be present, and the server check above only
    # covers one of them. On a machine with Ollama running but langchain not
    # installed, this used to fail with ModuleNotFoundError rather than skip -
    # a missing optional dependency reported as a broken test.
    pytest.importorskip("langchain_ollama", reason="langchain-ollama is not installed")

    from bfcl_data import load_category
    from harness import call_model_on_case
    from langchain_ollama import ChatOllama
    from models import get_chat_capable_models

    try:
        cases = load_category("simple_python")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not fetch real BFCL data: {exc}")

    models = get_chat_capable_models()
    if not models:
        pytest.skip("no chat-capable models pulled in local Ollama")

    chat = ChatOllama(
        model=models[0], temperature=0, keep_alive="2m", client_kwargs={"timeout": 90.0}
    )
    result = call_model_on_case(chat, cases[0])
    # Not asserting correctness (that's a measured outcome, not a test contract) --
    # just that the pipeline ran and produced a structurally valid result.
    assert result["error"] is None
    assert isinstance(result["predicted_calls"], list)
