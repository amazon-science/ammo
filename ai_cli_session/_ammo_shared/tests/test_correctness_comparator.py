# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Unit tests for Gate 5.1b v2 accuracy-based correctness comparator.

Tests use synthetic data (no model or GPU required).
"""
import sys
from pathlib import Path

# Ensure the scripts directory is importable regardless of the pytest invocation cwd.
# Matches the pattern used in tests/test_sweep_dp.py.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import pytest

from run_vllm_bench_latency_sweep import _compare_correctness


def _make_question(token_ids, logprobs=None):
    """Helper to build a synthetic question dict."""
    if logprobs is None:
        logprobs = []
    return {"token_ids": token_ids, "logprobs": logprobs, "text": "", "num_tokens": len(token_ids), "prompt_index": 0}


# ---- Test 1: opt_accuracy == baseline_accuracy → PASS ----

def test_equal_accuracy_pass():
    """Both get the same questions right → PASS."""
    golden = [_make_question([10, 20]), _make_question([30, 40])]
    opt = [_make_question([10, 20]), _make_question([30, 40])]
    labels = [42, 99]
    baseline_preds = [42, 99]  # both correct
    opt_preds = [42, 99]       # both correct
    result = _compare_correctness(
        golden_refs=golden, opt_outputs=opt,
        labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds,
    )
    assert result["verdict"] == "PASS"
    assert result["baseline_accuracy"] == result["optimized_accuracy"]
    assert result["accuracy_delta"] == 0.0


# ---- Test 2: opt_accuracy > baseline_accuracy → PASS ----

def test_higher_opt_accuracy_pass():
    """Optimized gets more questions right than baseline → PASS."""
    golden = [_make_question([10]), _make_question([20])]
    opt = [_make_question([10]), _make_question([20])]
    labels = [42, 99]
    baseline_preds = [42, 0]   # 1 correct
    opt_preds = [42, 99]       # 2 correct
    result = _compare_correctness(
        golden_refs=golden, opt_outputs=opt,
        labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds,
    )
    assert result["verdict"] == "PASS"
    assert result["optimized_accuracy"] > result["baseline_accuracy"]
    assert result["accuracy_delta"] > 0


# ---- Test 3: opt_accuracy < baseline_accuracy by 1 question → FAIL ----

def test_one_question_accuracy_drop_fail():
    """Optimized loses 1 question → FAIL."""
    golden = [_make_question([10]), _make_question([20])]
    opt = [_make_question([10]), _make_question([20])]
    labels = [42, 99]
    baseline_preds = [42, 99]  # 2 correct
    opt_preds = [42, 0]        # 1 correct
    result = _compare_correctness(
        golden_refs=golden, opt_outputs=opt,
        labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds,
    )
    assert result["verdict"] == "FAIL"
    assert result["accuracy_delta"] < 0


# ---- Test 4: opt_accuracy < baseline_accuracy by N questions, verify questions_lost ----

def test_multiple_questions_lost_fail():
    """Optimized loses multiple questions → FAIL with correct questions_lost."""
    n = 5
    golden = [_make_question([i]) for i in range(n)]
    opt = [_make_question([i]) for i in range(n)]
    labels = list(range(n))
    baseline_preds = list(range(n))        # all 5 correct
    opt_preds = [0, 1, 999, 999, 999]      # Q2, Q3, Q4 wrong
    result = _compare_correctness(
        golden_refs=golden, opt_outputs=opt,
        labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds,
    )
    assert result["verdict"] == "FAIL"
    assert result["questions_lost"] == [2, 3, 4]
    assert result["baseline_correct_count"] == 5
    assert result["optimized_correct_count"] == 2


# ---- Test 5: Question churn — lost 2, gained 2, same accuracy → PASS ----

def test_question_churn_same_accuracy_pass():
    """Lost 2, gained 2 → same accuracy → PASS."""
    n = 10
    golden = [_make_question([i]) for i in range(n)]
    opt = [_make_question([i]) for i in range(n)]
    labels = list(range(n))
    # Baseline gets Q0-Q6 correct (7 total)
    baseline_preds = [0, 1, 2, 3, 4, 5, 6, 999, 999, 999]
    # Opt gets Q0-Q4, Q7, Q8 correct (7 total) — lost Q5,Q6 gained Q7,Q8
    opt_preds = [0, 1, 2, 3, 4, 999, 999, 7, 8, 999]
    result = _compare_correctness(
        golden_refs=golden, opt_outputs=opt,
        labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds,
    )
    assert result["verdict"] == "PASS"
    assert result["questions_lost"] == [5, 6]
    assert result["questions_gained"] == [7, 8]
    assert result["baseline_accuracy"] == result["optimized_accuracy"]


# ---- Test 6: baseline_accuracy = 0 → infrastructure_error ----

def test_baseline_accuracy_zero_infrastructure_error():
    """Baseline gets 0 questions right → infrastructure_error flag."""
    golden = [_make_question([10]), _make_question([20])]
    opt = [_make_question([10]), _make_question([20])]
    labels = [42, 99]
    baseline_preds = [0, 0]    # 0 correct
    opt_preds = [42, 99]       # 2 correct
    result = _compare_correctness(
        golden_refs=golden, opt_outputs=opt,
        labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds,
    )
    assert result.get("infrastructure_error") is True
    assert result["verdict"] == "FAIL"
    assert result["baseline_correct_count"] == 0


# ---- Test 7: n = 0 (empty question list) → FAIL ----

def test_empty_questions_fail():
    """Zero questions → FAIL."""
    result = _compare_correctness(
        golden_refs=[], opt_outputs=[],
        labels=[], baseline_preds=[], opt_preds=[],
    )
    assert result["verdict"] == "FAIL"
    assert result["num_questions"] == 0


# ---- Test 8: All outputs empty (0 tokens) → FAIL ----

def test_all_empty_outputs_fail():
    """All questions produce 0 tokens → FAIL."""
    golden = [_make_question([]), _make_question([])]
    opt = [_make_question([]), _make_question([])]
    labels = [42, 99]
    baseline_preds = [42, 99]  # correct (from external scoring)
    opt_preds = [42, 99]       # correct (from external scoring)
    result = _compare_correctness(
        golden_refs=golden, opt_outputs=opt,
        labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds,
    )
    assert result["verdict"] == "FAIL"


# ---- Test 9: questions_lost and questions_gained mutually exclusive ----

def test_lost_gained_mutually_exclusive():
    """questions_lost and questions_gained have no overlap."""
    n = 6
    golden = [_make_question([i]) for i in range(n)]
    opt = [_make_question([i]) for i in range(n)]
    labels = list(range(n))
    # Baseline: Q0,Q1,Q2,Q3 correct
    baseline_preds = [0, 1, 2, 3, 999, 999]
    # Opt: Q0,Q1,Q4,Q5 correct (lost Q2,Q3; gained Q4,Q5)
    opt_preds = [0, 1, 999, 999, 4, 5]
    result = _compare_correctness(
        golden_refs=golden, opt_outputs=opt,
        labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds,
    )
    assert result["verdict"] == "PASS"
    lost_set = set(result["questions_lost"])
    gained_set = set(result["questions_gained"])
    assert lost_set & gained_set == set(), "questions_lost and questions_gained must be disjoint"
    assert result["questions_lost"] == [2, 3]
    assert result["questions_gained"] == [4, 5]


# ---- Test 10: Diagnostics populated but don't affect verdict → PASS ----

def test_diagnostics_dont_affect_verdict():
    """Token divergence in diagnostics but accuracy equal → PASS."""
    golden = [_make_question([10, 20, 30]), _make_question([40, 50, 60])]
    # Different tokens at every position
    opt = [_make_question([10, 99, 88]), _make_question([40, 50, 60])]
    labels = [42, 99]
    baseline_preds = [42, 99]
    opt_preds = [42, 99]
    result = _compare_correctness(
        golden_refs=golden, opt_outputs=opt,
        labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds,
    )
    assert result["verdict"] == "PASS"
    diag = result["diagnostics"]
    assert diag["divergent_questions"] == 1
    assert diag["first_divergence_positions_p50"] >= 0
    assert diag["first_divergence_positions_p95"] >= 0
    assert "churn_rate" in diag
    assert "note" in diag


# ---- Test 11: accuracy_delta math correct ----

def test_accuracy_delta_math():
    """Verify accuracy_delta = optimized_accuracy - baseline_accuracy."""
    n = 4
    golden = [_make_question([i]) for i in range(n)]
    opt = [_make_question([i]) for i in range(n)]
    labels = [0, 1, 2, 3]
    baseline_preds = [0, 1, 2, 3]  # 4/4 = 1.0
    opt_preds = [0, 1, 2, 999]     # 3/4 = 0.75
    result = _compare_correctness(
        golden_refs=golden, opt_outputs=opt,
        labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds,
    )
    assert result["baseline_accuracy"] == 1.0
    assert result["optimized_accuracy"] == 0.75
    assert result["accuracy_delta"] == -0.25
    assert result["verdict"] == "FAIL"


# ---- Test 12: Large n (200 questions) with 1-question accuracy drop → FAIL ----

def test_large_n_one_question_drop_fail():
    """200 questions, optimized loses exactly 1 → FAIL under strict tolerance.

    NOTE: The new default tolerance (1.0pp) would allow a 1/200 drop (0.5pp).
    This test asserts the strict-mode semantics via ``tolerance_pct=0.0``.
    """
    n = 200
    golden = [_make_question([i]) for i in range(n)]
    opt = [_make_question([i]) for i in range(n)]
    labels = list(range(n))
    baseline_preds = list(range(n))  # all 200 correct
    opt_preds = list(range(n))
    opt_preds[99] = -1  # Q99 wrong
    result = _compare_correctness(
        golden_refs=golden, opt_outputs=opt,
        labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds,
        tolerance_pct=0.0,
    )
    assert result["verdict"] == "FAIL"
    assert result["num_questions"] == 200
    assert result["baseline_correct_count"] == 200
    assert result["optimized_correct_count"] == 199
    assert result["questions_lost"] == [99]
    assert result["questions_gained"] == []
    assert result["accuracy_delta"] < 0


# ---- Tolerance tests (Gate 5.1b configurable tolerance_pct) ----

def _inputs(n, b_correct, o_correct):
    """Build synthetic inputs with N total, B baseline-correct, O opt-correct."""
    golden = [_make_question([i + 1]) for i in range(n)]
    opt = [_make_question([i + 1]) for i in range(n)]
    labels = list(range(n))
    baseline_preds = [i if i < b_correct else -1 for i in range(n)]
    opt_preds = [i if i < o_correct else -1 for i in range(n)]
    return dict(
        golden_refs=golden, opt_outputs=opt,
        labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds,
    )


def test_tolerance_pass_at_exact_boundary():
    """n=200, baseline=100%, opt=99%, tol=1.0pp → PASS (boundary inclusive)."""
    r = _compare_correctness(tolerance_pct=1.0, **_inputs(200, 200, 198))
    assert r["verdict"] == "PASS"
    assert r["threshold"] == round(1.0 - 0.01, 4)


def test_tolerance_fail_below_boundary():
    """n=200, baseline=100%, opt=98%, tol=1.0pp → FAIL."""
    r = _compare_correctness(tolerance_pct=1.0, **_inputs(200, 200, 196))
    assert r["verdict"] == "FAIL"


def test_tolerance_zero_is_strict():
    """tol=0 → strict comparison (opt >= baseline)."""
    r = _compare_correctness(tolerance_pct=0.0, **_inputs(200, 200, 199))
    assert r["verdict"] == "FAIL"


def test_tolerance_default_is_one_pp():
    """Omit tolerance_pct kwarg → default 1.0pp → baseline 89%, opt 88% passes."""
    r = _compare_correctness(**_inputs(100, 89, 88))
    assert r["verdict"] == "PASS"
    assert r["tolerance_pct"] == 1.0


def test_tolerance_opt_above_baseline_passes_any_tolerance():
    """opt >= baseline → PASS regardless of tolerance (upper bound unchanged)."""
    for tol in (0.0, 1.0, 5.0):
        r = _compare_correctness(tolerance_pct=tol, **_inputs(10, 8, 9))
        assert r["verdict"] == "PASS"


def test_tolerance_pct_and_threshold_in_return_dict():
    r = _compare_correctness(tolerance_pct=2.5, **_inputs(100, 80, 80))
    assert r["tolerance_pct"] == 2.5
    assert r["threshold"] == round(0.80 - 0.025, 4)


def test_message_pass_format():
    r = _compare_correctness(tolerance_pct=1.0, **_inputs(200, 200, 198))
    msg = r["message"]
    assert "PASS:" in msg
    assert "99.0%" in msg
    assert "threshold" in msg
    assert "tolerance" in msg


def test_message_fail_format():
    r = _compare_correctness(tolerance_pct=1.0, **_inputs(200, 200, 196))
    msg = r["message"]
    assert "FAIL:" in msg
    assert "(196/200)" in msg
    assert "threshold" in msg
    assert "1.0pp tolerance" in msg


def test_infrastructure_error_overrides_tolerance():
    """baseline=0%, tol=100 → still infra error."""
    r = _compare_correctness(tolerance_pct=100.0, **_inputs(10, 0, 5))
    assert r.get("infrastructure_error") is True
    assert r["verdict"] == "FAIL"
