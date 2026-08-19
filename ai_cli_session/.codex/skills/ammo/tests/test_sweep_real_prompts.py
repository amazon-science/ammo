#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for real-prompt timing in run_vllm_bench_latency_sweep.py.

Covers the pure helper functions added by the 2026-06-22 real-prompt timing
design (windowed bundled-GSM8K timing prompts + additive spec_decode block):

  - _timing_prompt_seed         : PYTHONHASHSEED-independent crc32 seed
  - _build_timing_prompts       : per-bucket windowing (gsm8k) + legacy (random)
  - _load_timing_corpus_text    : full-file corpus (NOT the 200-item subset)
  - _load_timing_corpus_tokens  : tokenize-once-per-name caching
  - _spec_decode_block_from_diff: acceptance diff math + independent guards
  - _build_label_result_entry   : spec_decode transport (present / omit-not-null)
  - main() parser default       : --dummy-prompt-source defaults to "gsm8k"

These tests do NOT require a GPU or importing vLLM — every function exercised
here is pure (numpy is injected; corpus + tokenizer are stubbed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the scripts directory is importable (mirrors test_sweep_dp.py).
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from run_vllm_bench_latency_sweep import (  # noqa: E402
    _GSM8K_FULL_PATH,
    _TIMING_CORPUS_CACHE,
    _build_label_result_entry,
    _build_timing_prompts,
    _load_timing_corpus_text,
    _load_timing_corpus_tokens,
    _metrics_from_vllm_latency_json,
    _spec_decode_block_from_diff,
    _timing_prompt_seed,
)


# ---------------------------------------------------------------------------
# Stub tokenizer: whitespace-splitting encode() with a call counter. GPU-free.
# ---------------------------------------------------------------------------
class _StubTokenizer:
    def __init__(self, name="stub-tok"):
        self.name_or_path = name
        self.encode_calls = 0

    def encode(self, text, add_special_tokens=False):
        # add_special_tokens accepted but ignored; splits on whitespace and maps
        # each token to a small deterministic int so windows are real "IDs".
        self.encode_calls += 1
        toks = text.split()
        return [(len(t) * 31 + ord(t[0])) % 50000 for t in toks if t]


class _NoKwargTokenizer:
    """encode() without add_special_tokens -> exercises the TypeError fallback."""

    def __init__(self, name="nokwarg-tok"):
        self.name_or_path = name
        self.encode_calls = 0

    def encode(self, text):
        self.encode_calls += 1
        return [hash(t) % 1000 for t in text.split() if t]


# Tiny synthetic corpus used by the windowing tests (independent of real data).
_TINY_CORPUS = [10, 11, 12, 13, 14, 15, 16]  # n = 7


# ---------------------------------------------------------------------------
# 1. Window length exactly input_len, including input_len > corpus (modulo wrap)
# ---------------------------------------------------------------------------
class TestWindowLength:
    def test_short_window(self):
        prompts = _build_timing_prompts(
            _TINY_CORPUS, batch_size=4, input_len=3, output_len=8, np_module=np
        )
        assert len(prompts) == 4
        for p in prompts:
            assert len(p["prompt_token_ids"]) == 3

    def test_window_longer_than_corpus_wraps(self):
        # input_len (20) > corpus length (7) -> must still return exactly 20.
        prompts = _build_timing_prompts(
            _TINY_CORPUS, batch_size=2, input_len=20, output_len=4, np_module=np
        )
        assert len(prompts) == 2
        for p in prompts:
            ids = p["prompt_token_ids"]
            assert len(ids) == 20
            # Verify it actually wraps with exact modulo form.
            start = None
            # Recover start by matching the first id to a corpus position; then
            # confirm the full window equals the modulo slice from that start.
            for cand in range(len(_TINY_CORPUS)):
                window = [_TINY_CORPUS[(cand + k) % len(_TINY_CORPUS)] for k in range(20)]
                if window == ids:
                    start = cand
                    break
            assert start is not None, "window is not a contiguous modulo slice"


# ---------------------------------------------------------------------------
# 2. All IDs come from the corpus (membership) for gsm8k source
# ---------------------------------------------------------------------------
class TestMembership:
    def test_all_ids_in_corpus(self):
        corpus_set = set(_TINY_CORPUS)
        prompts = _build_timing_prompts(
            _TINY_CORPUS, batch_size=6, input_len=15, output_len=4, np_module=np
        )
        for p in prompts:
            for tid in p["prompt_token_ids"]:
                assert tid in corpus_set


# ---------------------------------------------------------------------------
# 3. Determinism: identical windows across two calls; crc32 seed stable
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_same_args_identical(self):
        a = _build_timing_prompts(
            _TINY_CORPUS, batch_size=5, input_len=9, output_len=16, np_module=np
        )
        b = _build_timing_prompts(
            _TINY_CORPUS, batch_size=5, input_len=9, output_len=16, np_module=np
        )
        assert a == b

    def test_seed_is_stable_value(self):
        # crc32 over the canonical string is a fixed value (not PYTHONHASHSEED).
        # Recompute the expected value with the same canonical form.
        import zlib
        expected = zlib.crc32(b"64:512:8")
        assert _timing_prompt_seed(64, 512, 8) == expected
        # And stable across repeated calls.
        assert _timing_prompt_seed(64, 512, 8) == _timing_prompt_seed(64, 512, 8)

    def test_different_workloads_differ(self):
        a = _build_timing_prompts(
            _TINY_CORPUS, batch_size=4, input_len=6, output_len=8, np_module=np
        )
        b = _build_timing_prompts(
            _TINY_CORPUS, batch_size=4, input_len=6, output_len=16, np_module=np
        )
        # Different output_len feeds the seed -> different windows (very likely).
        assert a != b


# ---------------------------------------------------------------------------
# 4. A/B parity: two separate builds with same args -> identical
# ---------------------------------------------------------------------------
class TestABParity:
    def test_baseline_opt_identical(self):
        # Simulate two arms (baseline + opt) with the SAME workload.
        baseline = _build_timing_prompts(
            _TINY_CORPUS, batch_size=8, input_len=12, output_len=64, np_module=np
        )
        opt = _build_timing_prompts(
            _TINY_CORPUS, batch_size=8, input_len=12, output_len=64, np_module=np
        )
        assert baseline == opt


# ---------------------------------------------------------------------------
# 5. source="random": shape (bs, input_len); differs from gsm8k for same args
# ---------------------------------------------------------------------------
class TestRandomSource:
    def test_random_shape(self):
        prompts = _build_timing_prompts(
            None, batch_size=5, input_len=7, output_len=4, source="random", np_module=np
        )
        assert len(prompts) == 5
        for p in prompts:
            assert len(p["prompt_token_ids"]) == 7
            for tid in p["prompt_token_ids"]:
                assert 0 <= tid < 10000

    def test_random_differs_from_gsm8k(self):
        # Use the SAME corpus for both calls so the difference is attributable
        # to the windowing path, NOT to range disjointness (random ignores the
        # corpus and emits randint(10000); a contiguous gsm8k modulo-slice of
        # _TINY_CORPUS ⊂ [10,16] can only equal a random draw by astronomical
        # coincidence — but more importantly the gsm8k output MUST be a valid
        # modulo slice, which a broken branch would not produce).
        gsm = _build_timing_prompts(
            _TINY_CORPUS, batch_size=4, input_len=8, output_len=4,
            source="gsm8k", np_module=np,
        )
        rand = _build_timing_prompts(
            _TINY_CORPUS, batch_size=4, input_len=8, output_len=4,
            source="random", np_module=np,
        )
        assert gsm != rand
        # Strong positive check: every gsm8k window is a contiguous modulo slice
        # of the corpus (so all ids ∈ corpus); random windows are not bounded by
        # the corpus and will contain ids outside [10,16].
        corpus_set = set(_TINY_CORPUS)
        for p in gsm:
            assert set(p["prompt_token_ids"]) <= corpus_set
        assert any(
            any(tid not in corpus_set for tid in p["prompt_token_ids"])
            for p in rand
        )

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError):
            _build_timing_prompts(
                _TINY_CORPUS, batch_size=1, input_len=4, output_len=4,
                source="bogus", np_module=np,
            )


# ---------------------------------------------------------------------------
# 6. _spec_decode_block_from_diff: rate/mean_len + independent guards
# ---------------------------------------------------------------------------
class TestSpecDecodeDiff:
    def test_normal_diff(self):
        before = {"num_drafts": 10, "num_draft_tokens": 40, "num_accepted_tokens": 20}
        after = {"num_drafts": 30, "num_draft_tokens": 120, "num_accepted_tokens": 90}
        block = _spec_decode_block_from_diff(before, after)
        assert block is not None
        # drafts=20, draft_tokens=80, accepted=70
        assert block["num_drafts"] == 20
        assert block["num_draft_tokens"] == 80
        assert block["num_accepted_tokens"] == 70
        assert block["draft_acceptance_rate"] == pytest.approx(70 / 80)
        assert block["mean_acceptance_length"] == pytest.approx(1.0 + 70 / 20)

    def test_drafts_zero_omits_mean_len_keeps_rate(self):
        # num_drafts diff == 0 but draft_tokens > 0 -> rate kept, mean_len omitted.
        before = {"num_drafts": 5, "num_draft_tokens": 10, "num_accepted_tokens": 4}
        after = {"num_drafts": 5, "num_draft_tokens": 18, "num_accepted_tokens": 10}
        block = _spec_decode_block_from_diff(before, after)
        assert block is not None
        assert "mean_acceptance_length" not in block
        assert "draft_acceptance_rate" in block
        assert block["num_drafts"] == 0
        assert block["draft_acceptance_rate"] == pytest.approx(6 / 8)

    def test_both_zero_returns_none(self):
        before = {"num_drafts": 5, "num_draft_tokens": 10, "num_accepted_tokens": 4}
        after = {"num_drafts": 5, "num_draft_tokens": 10, "num_accepted_tokens": 4}
        assert _spec_decode_block_from_diff(before, after) is None

    def test_draft_tokens_zero_omits_rate_keeps_mean_len(self):
        # drafts > 0 but draft_tokens diff == 0 -> rate omitted, mean_len kept.
        before = {"num_drafts": 1, "num_draft_tokens": 7, "num_accepted_tokens": 1}
        after = {"num_drafts": 4, "num_draft_tokens": 7, "num_accepted_tokens": 4}
        block = _spec_decode_block_from_diff(before, after)
        assert block is not None
        assert "draft_acceptance_rate" not in block
        assert "mean_acceptance_length" in block
        assert block["num_drafts"] == 3
        assert block["mean_acceptance_length"] == pytest.approx(1.0 + 3 / 3)

    def test_none_inputs_return_none(self):
        assert _spec_decode_block_from_diff(None, None) is None
        assert _spec_decode_block_from_diff(None, {"num_drafts": 1}) is None
        assert _spec_decode_block_from_diff({"num_drafts": 1}, None) is None


# ---------------------------------------------------------------------------
# 7. Default guard: main()'s --dummy-prompt-source defaults to "gsm8k"
# ---------------------------------------------------------------------------
class TestDefaultGuard:
    def test_parser_default_is_gsm8k(self, monkeypatch):
        import argparse as _argparse
        import run_vllm_bench_latency_sweep as mod

        captured = {}

        class _Stop(Exception):
            pass

        real_parse = _argparse.ArgumentParser.parse_args

        def _capture(self, *a, **kw):
            ns = real_parse(self, *a, **kw)
            captured["ns"] = ns
            raise _Stop()

        monkeypatch.setattr(_argparse.ArgumentParser, "parse_args", _capture)
        # Minimal valid argv: only --artifact-dir is required.
        monkeypatch.setattr(sys, "argv", ["prog", "--artifact-dir", "/tmp/does-not-matter"])
        with pytest.raises(_Stop):
            mod.main()
        assert captured["ns"].dummy_prompt_source == "gsm8k"

    def test_parser_accepts_random(self, monkeypatch):
        import argparse as _argparse
        import run_vllm_bench_latency_sweep as mod

        captured = {}

        class _Stop(Exception):
            pass

        real_parse = _argparse.ArgumentParser.parse_args

        def _capture(self, *a, **kw):
            ns = real_parse(self, *a, **kw)
            captured["ns"] = ns
            raise _Stop()

        monkeypatch.setattr(_argparse.ArgumentParser, "parse_args", _capture)
        monkeypatch.setattr(
            sys, "argv",
            ["prog", "--artifact-dir", "/tmp/x", "--dummy-prompt-source", "random"],
        )
        with pytest.raises(_Stop):
            mod.main()
        assert captured["ns"].dummy_prompt_source == "random"


# ---------------------------------------------------------------------------
# 7b. Startup fail-fast precheck (child-aware): bare-file 'random' child must
#     NOT fail-FAST on the precheck when the bundled data is absent; a 'gsm8k'
#     run/child MUST. We detect by the precheck's unique SystemExit message,
#     forcing _GSM8K_FULL_PATH.exists() -> False and never touching a GPU.
# ---------------------------------------------------------------------------
_PRECHECK_MARKER = "Real-prompt timing requires the bundled GSM8K full dataset"


class TestStartupPrecheck:
    def _drive(self, monkeypatch, argv):
        """Run main() with the bundled data forced ABSENT.

        Returns the SystemExit message string. The precheck's SystemExit carries
        a unique marker; any LATER SystemExit (e.g. missing target.json once the
        precheck has PASSED) will not, which is exactly how we distinguish a
        precheck failure from the precheck passing.
        """
        import run_vllm_bench_latency_sweep as mod

        path_cls = type(mod._GSM8K_FULL_PATH)
        real_exists = path_cls.exists

        def _patched_exists(self):
            # Force only the GSM8K full path to report absent; other Paths real.
            if self == mod._GSM8K_FULL_PATH:
                return False
            return real_exists(self)

        monkeypatch.setattr(path_cls, "exists", _patched_exists)
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exc:
            mod.main()
        return str(exc.value)

    def test_gsm8k_parent_fails_fast(self, monkeypatch):
        msg = self._drive(monkeypatch, ["prog", "--artifact-dir", "/tmp/x"])
        assert _PRECHECK_MARKER in msg

    def test_gsm8k_child_fails_fast(self, monkeypatch):
        msg = self._drive(
            monkeypatch,
            ["prog", "--artifact-dir", "/tmp/x", "--_child-label", "baseline",
             "--_dummy-prompt-source", "gsm8k"],
        )
        assert _PRECHECK_MARKER in msg

    def test_random_child_skips_precheck(self, monkeypatch):
        # A random child must NOT trip the precheck even with data absent. It
        # will still SystemExit later (no target.json), but WITHOUT the marker.
        msg = self._drive(
            monkeypatch,
            ["prog", "--artifact-dir", "/tmp/x", "--_child-label", "baseline",
             "--_dummy-prompt-source", "random"],
        )
        assert _PRECHECK_MARKER not in msg

    def test_random_parent_skips_precheck(self, monkeypatch):
        msg = self._drive(
            monkeypatch,
            ["prog", "--artifact-dir", "/tmp/x", "--dummy-prompt-source", "random"],
        )
        assert _PRECHECK_MARKER not in msg


# ---------------------------------------------------------------------------
# 8. Corpus floor: full file (1319+5), not subset; long text, many questions
# ---------------------------------------------------------------------------
class TestCorpusFloor:
    def test_text_is_full_file(self):
        text = _load_timing_corpus_text()
        # Full file is ~177K tokens; the text should be hundreds of KB.
        assert len(text) > 200_000
        # Every item contributes a "Question:" prefix; the full split has
        # 1319 test + 5 train = 1324 items, so > 1000 occurrences.
        assert text.count("Question:") > 1000

    def test_token_floor_with_whitespace_stub(self):
        # Clear cache so this stub actually tokenizes the full corpus.
        _TIMING_CORPUS_CACHE.clear()
        tok = _StubTokenizer(name="floor-stub")
        ids = _load_timing_corpus_tokens(tok)
        # A whitespace split of the full corpus yields well over 100K tokens.
        assert len(ids) > 100_000
        # Sanity: real data file is present (precondition for this test).
        assert _GSM8K_FULL_PATH.exists()


# ---------------------------------------------------------------------------
# 9. Cache reuse: two loads with same tokenizer object call .encode once
# ---------------------------------------------------------------------------
class TestCacheReuse:
    def test_encode_called_once(self):
        _TIMING_CORPUS_CACHE.clear()
        tok = _StubTokenizer(name="cache-stub")
        first = _load_timing_corpus_tokens(tok)
        second = _load_timing_corpus_tokens(tok)
        assert first is second  # same cached list object
        assert tok.encode_calls == 1

    def test_typeerror_fallback_tokenizer(self):
        _TIMING_CORPUS_CACHE.clear()
        tok = _NoKwargTokenizer(name="nokwarg")
        ids = _load_timing_corpus_tokens(tok)
        assert len(ids) > 0
        assert tok.encode_calls == 1

    def test_missing_tokenizer_raises(self):
        with pytest.raises(RuntimeError):
            _load_timing_corpus_tokens(None)


# ---------------------------------------------------------------------------
# 10. Transport: _build_label_result_entry carries / omits spec_decode
# ---------------------------------------------------------------------------
class TestSpecDecodeTransport:
    def _entry(self, spec):
        return _build_label_result_entry(
            cmd=["x"],
            env_overrides={},
            metrics={"avg_s": 1.0},
            log_rel="logs/x.log",
            output_json_rel="json/x.json",
            runner_json_rel="json/x.runner.json",
            ok=True,
            returncode=0,
            evidence_status="ok",
            evidence={},
            timing={},
            spec_decode=spec,
        )

    def test_spec_decode_present(self):
        block = {
            "num_drafts": 5,
            "num_draft_tokens": 20,
            "num_accepted_tokens": 14,
            "draft_acceptance_rate": 0.7,
            "mean_acceptance_length": 3.8,
        }
        entry = self._entry(block)
        assert entry["spec_decode"] == block

    def test_spec_decode_absent_when_none(self):
        entry = self._entry(None)
        assert "spec_decode" not in entry

    def test_float_converter_drops_nested_spec_decode(self):
        # [MF1] Locks the "lift directly, NOT via _metrics_from_vllm_latency_json"
        # requirement: that converter is a float-only allowlist, so routing the
        # raw child JSON through it would silently DROP the nested spec_decode
        # dict. A future refactor that re-routes spec_decode through it would
        # regress to the original transport bug — this test catches that.
        raw = {
            "avg_latency": 1.0,
            "decode_avg_s": 0.5,
            "spec_decode": {"num_drafts": 5, "num_accepted_tokens": 14},
        }
        converted = _metrics_from_vllm_latency_json(raw)
        assert "spec_decode" not in converted
        # Sanity: the scalar metrics it IS responsible for still come through.
        assert converted.get("avg_s") == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
