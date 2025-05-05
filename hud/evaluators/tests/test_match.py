from __future__ import annotations

import pytest

from hud.evaluators.match import match_all, match_diff, match_fuzzy, match_regex, match_single


@pytest.mark.parametrize(
    "response, answer, expected_score, expected_reason, expected_mode",
    [
        ("Hello, world!", "world", 1.0, "Exact match", "single"),
        ("Hello, world!", "not world", 0.0, "No exact match found", "single"),
    ],
)
def test_match_single(
    response: str,
    answer: str,
    expected_score: float,
    expected_reason: str,
    expected_mode: str,
):
    result = match_single(response, answer)
    assert result.score == expected_score
    assert result.reason == expected_reason
    assert result.mode == expected_mode


@pytest.mark.parametrize(
    "response, answers, expected_score, expected_reason, expected_mode",
    [
        ("Hello, world!", ["world", "hello"], 1.0, "All 2 expected items found", "all"),
        ("Hello, world!", ["world", "not hello"], 0.5, "Only 1 of 2 expected items found", "all"),
    ],
)
def test_match_all(
    response: str,
    answers: list[str],
    expected_score: float,
    expected_reason: str,
    expected_mode: str,
):
    result = match_all(response, answers)
    assert result.score == expected_score
    assert result.reason == expected_reason
    assert result.mode == expected_mode


@pytest.mark.parametrize(
    "response, answer, expected_score, expected_reason, expected_mode",
    [
        ("hello world", "hello world", 1.0, "Fuzzy match with 100.0% similarity", "fuzz"),
        ("hello wrld", "hello world", 0.9, "Fuzzy match with 90.9% similarity", "fuzz"),
        ("hello", "hello world", 0.45, "Fuzzy match with 45.5% similarity", "fuzz"),
        ("", "hello world", 0.0, "Fuzzy match with 0.0% similarity", "fuzz"),
    ],
)
def test_match_fuzzy(
    response: str,
    answer: str,
    expected_score: float,
    expected_reason: str,
    expected_mode: str,
):
    result = match_fuzzy(response, answer)
    assert result.score == pytest.approx(expected_score, abs=1e-2)
    assert result.reason == expected_reason
    assert result.mode == expected_mode


@pytest.mark.parametrize(
    "response, pattern, expected_score, expected_reason, expected_mode",
    [
        ("hello world", r"hello.*", 1.0, "Regex pattern matched", "regex"),
        ("hello world", r"^hello.*$", 1.0, "Regex pattern matched", "regex"),
        ("hello world", r"world$", 1.0, "Regex pattern matched", "regex"),
        ("hello world", r"^goodbye.*$", 0.0, "Regex pattern did not match", "regex"),
        ("hello world", r"[invalid[", 0.0, "Invalid regex pattern", "regex"),
    ],
)
def test_match_regex(
    response: str,
    pattern: str,
    expected_score: float,
    expected_reason: str,
    expected_mode: str,
):
    result = match_regex(response, pattern)
    assert result.score == expected_score
    assert result.reason == expected_reason
    assert result.mode == expected_mode


@pytest.mark.parametrize(
    "response, answer, expected_score, expected_reason, expected_mode",
    [
        ("hello world", "hello world", 1.0, "String difference with 100.0% similarity", "diff"),
        ("hello", "hello world", 0.625, "String difference with 62.5% similarity", "diff"),
        ("", "hello world", 0.0, "String difference with 0.0% similarity", "diff"),
        (100, 100, 1.0, "Numeric difference: 0", "diff"),
        (90, 100, 0.9, "Numeric difference: 10", "diff"),
        (0, 100, 0.0, "Numeric difference: 100", "diff"),
        (-100, 100, 0.0, "Numeric difference: 200", "diff"),
    ],
)
def test_match_diff(
    response: str | int | float,
    answer: str | int | float,
    expected_score: float,
    expected_reason: str,
    expected_mode: str,
):
    result = match_diff(response, answer)
    assert result.score == pytest.approx(expected_score, abs=1e-2)
    assert result.reason == expected_reason
    assert result.mode == expected_mode
