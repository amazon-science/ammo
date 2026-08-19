# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for mine_trace.py — the sole owner of every mined Stage-2 number.

Fixtures are SYNTHETIC CUPTI sqlite tables built in-test (a real nsys export is
~750 KB and is never committed). Each fixture carries only the columns the
script requires, so a missing-column regression fails here.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mine_trace.py"
_spec = importlib.util.spec_from_file_location("mine_trace", str(_SCRIPT))
mine_trace = importlib.util.module_from_spec(_spec)
sys.modules["mine_trace"] = mine_trace
_spec.loader.exec_module(mine_trace)


ALL_COLUMNS = (
    "start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, "
    "globalPid INTEGER, demangledName INTEGER, graphId INTEGER"
)


def write_sqlite(
    path: Path,
    kernels,
    table: str = "CUPTI_ACTIVITY_KIND_KERNEL",
    columns: str = ALL_COLUMNS,
    meta: dict | None = None,
):
    """Build a minimal CUPTI-shaped export.

    `kernels` items are (start, end, symbol) or (start, end, symbol, device_id,
    global_pid, graph_id).
    """
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    con.execute(f"CREATE TABLE {table} ({columns})")
    con.execute("CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT)")
    defaults = {
        "EXPORT_PRODUCT_VERSION": "2026.3.1.157",
        "EXPORT_SCHEMA_VERSION": "3.27.0",
    }
    defaults.update(meta or {})
    con.executemany(
        "INSERT INTO META_DATA_EXPORT (name, value) VALUES (?, ?)", sorted(defaults.items())
    )
    symbols: dict[str, int] = {}
    field_names = [c.split()[0] for c in columns.split(", ")]
    for kernel in kernels:
        start, end, symbol = kernel[0], kernel[1], kernel[2]
        device_id = kernel[3] if len(kernel) > 3 else 0
        global_pid = kernel[4] if len(kernel) > 4 else 111
        graph_id = kernel[5] if len(kernel) > 5 else 1
        if symbol not in symbols:
            symbols[symbol] = len(symbols) + 1
            con.execute(
                "INSERT INTO StringIds (id, value) VALUES (?, ?)", (symbols[symbol], symbol)
            )
        values = {
            "start": start,
            "end": end,
            "deviceId": device_id,
            "streamId": 7,
            "globalPid": global_pid,
            "demangledName": symbols[symbol],
            "graphId": graph_id,
        }
        present = [f for f in field_names if f in values]
        con.execute(
            f"INSERT INTO {table} ({', '.join(present)}) "
            f"VALUES ({', '.join('?' for _ in present)})",
            [values[f] for f in present],
        )
    con.commit()
    con.close()


def transient_plus_step(step_kernels, gap_ns=1_000_000):
    """Prefix a 2-kernel capture-window transient separated by one big gap."""
    prefix = [(0, 1_000, "transient_setup"), (2_000, 3_000, "transient_setup")]
    shift = 3_000 + gap_ns
    return prefix + [
        (k[0] + shift, k[1] + shift) + tuple(k[2:]) for k in step_kernels
    ]


def sweep_json(
    bs=32,
    avg_s=10.0,
    decode_avg_s=9.0,
    prefill_avg_s=0.5,
    output_len=512,
    tpot_s=None,
    stored_share=None,
    extra=None,
):
    metrics = {
        "avg_s": avg_s,
        "decode_avg_s": decode_avg_s,
        "prefill_avg_s": prefill_avg_s,
    }
    if tpot_s is not None:
        metrics["tpot_s"] = tpot_s
    if stored_share is not None:
        metrics["decode_share_of_e2e"] = stored_share
    metrics.update(extra or {})
    return {
        "workload": {"num_iters": 10, "output_len": output_len},
        "results": [{"batch_size": bs, "output_len": output_len, "baseline": metrics}],
    }


def write_config(tmp_path: Path, config: dict, sweep: dict) -> Path:
    (tmp_path / "e2e_latency_results.json").write_text(json.dumps(sweep), encoding="utf-8")
    config.setdefault("round", 1)
    config.setdefault("artifact_dir", str(tmp_path))
    config.setdefault("arm", "baseline")
    config.setdefault("e2e_results", "e2e_latency_results.json")
    config.setdefault("segmentation", {"policy": "max_gap"})
    path = tmp_path / "mine_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def run(tmp_path: Path, config_path: Path, capsys=None):
    """Invoke main(); return (rc, mined_or_None, tables_or_None)."""
    out_json = tmp_path / "mined.json"
    out_md = tmp_path / "tables.md"
    rc = mine_trace.main(
        ["--config", str(config_path), "--out-json", str(out_json), "--out-md", str(out_md)]
    )
    mined = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else None
    tables = out_md.read_text(encoding="utf-8") if out_md.exists() else None
    return rc, mined, tables


# --- 1. union arithmetic never sums concurrent kernels -----------------------


def test_merged_busy_unions_a_constructed_overlap():
    # 100..200 and 150..300 overlap by 50: union 200, sum 250.
    intervals = [(100, 200), (150, 300), (400, 500)]
    assert mine_trace.merged_busy(intervals) == 300
    assert mine_trace.duration_sum(intervals) == 350


def test_overlapping_kernels_report_union_not_sum(tmp_path):
    # Two concurrent streams: 0..1000 and 500..1500 → union 1500, sum 2000.
    step = [
        (0, 1_000, "target_kernel", 0, 111, 1),
        (500, 1_500, "target_kernel", 0, 222, 1),
    ]
    write_sqlite(tmp_path / "t.sqlite", transient_plus_step(step))
    config = write_config(
        tmp_path,
        {
            "traces": [{"bs": 32, "sqlite": "t.sqlite"}],
            "families": [{"label": "target", "symbol_patterns": ["target_kernel"]}],
        },
        sweep_json(decode_avg_s=1.5e-6 * 511, tpot_s=1.5e-6),
    )
    rc, mined, _ = run(tmp_path, config)
    assert rc == 0
    depth = mined["per_bs"][0]["depths"][0]
    assert depth["merged_busy_ns"] == 1_500
    assert depth["sum_ns"] == 2_000
    assert depth["overlap_ns"] == 500
    assert mined["per_bs"][0]["mean_merged_busy_per_step_s"] == pytest.approx(1.5e-6)


# --- 2. segmentation determinism --------------------------------------------


def test_max_gap_segmentation_is_deterministic_and_reports_separation(tmp_path):
    step = [(0, 1_000, "k"), (1_100, 2_000, "k")]
    write_sqlite(tmp_path / "t.sqlite", transient_plus_step(step, gap_ns=500_000))
    config = write_config(
        tmp_path,
        {
            "traces": [{"bs": 32, "sqlite": "t.sqlite"}],
            "families": [{"label": "k", "symbol_patterns": ["^k$"]}],
        },
        sweep_json(decode_avg_s=1.9e-6 * 511, tpot_s=1.9e-6),
    )
    first = run(tmp_path, config)[1]["per_bs"][0]["depths"][0]
    second = run(tmp_path, config)[1]["per_bs"][0]["depths"][0]
    assert first["delimiter"] == second["delimiter"]
    assert first["delimiter"]["policy"] == "max_gap"
    assert first["delimiter"]["gap_ns"] == 500_000
    assert first["delimiter"]["at_index"] == 2
    assert first["delimiter"]["first_kernel_after"] == "k"
    # next gap is the 1000 ns intra-transient gap → separation 500x
    assert first["delimiter"]["separation_ratio"] == pytest.approx(500.0)
    assert first["n_kernels_window"] == 2


def test_delimiter_symbol_policy_segments_on_last_occurrence(tmp_path):
    kernels = [
        (0, 100, "warm"),
        (200, 300, "step_marker"),
        (400, 500, "k"),
        (600, 700, "k"),
    ]
    write_sqlite(tmp_path / "t.sqlite", kernels)
    config = write_config(
        tmp_path,
        {
            "traces": [{"bs": 32, "sqlite": "t.sqlite"}],
            "segmentation": {"policy": "delimiter_symbol", "symbol": "step_marker"},
            "families": [
                {"label": "marker", "symbol_patterns": ["step_marker"]},
                {"label": "k", "symbol_patterns": ["^k$"]},
            ],
        },
        sweep_json(decode_avg_s=3e-7 * 511, tpot_s=3e-7),
    )
    rc, mined, _ = run(tmp_path, config)
    assert rc == 0
    depth = mined["per_bs"][0]["depths"][0]
    assert depth["delimiter"]["policy"] == "delimiter_symbol"
    assert depth["delimiter"]["occurrences"] == 1
    assert depth["n_kernels_window"] == 3


def test_low_separation_ratio_warns_without_failing(tmp_path):
    # Two comparable gaps → separation_ratio 1.25, below the default floor of 2.
    kernels = [
        (0, 100, "a"),
        (1_100, 1_200, "a"),
        (2_450, 2_550, "k"),
        (2_650, 2_750, "k"),
    ]
    write_sqlite(tmp_path / "t.sqlite", kernels)
    config = write_config(
        tmp_path,
        {
            "traces": [{"bs": 32, "sqlite": "t.sqlite"}],
            "families": [
                {"label": "a", "symbol_patterns": ["^a$"]},
                {"label": "k", "symbol_patterns": ["^k$"]},
            ],
        },
        sweep_json(decode_avg_s=3e-7 * 511, tpot_s=3e-7),
    )
    rc, mined, _ = run(tmp_path, config)
    assert rc == 0
    assert any("separation_ratio" in w for w in mined["warnings"])


# --- 3. denominator re-derivation ignores the stored share ------------------


def test_shares_are_rederived_and_stored_phase_sum_warns(tmp_path):
    # 17.437 ms/step x 511 steps ~= the 9 s decode wall → decode_busy ~0.99.
    step = [(0, 17_437_000, "k")]
    write_sqlite(tmp_path / "t.sqlite", transient_plus_step(step, gap_ns=20_000_000))
    avg_s, decode_avg_s, prefill_avg_s = 10.0, 9.0, 0.5
    phase_sum = decode_avg_s / (prefill_avg_s + decode_avg_s)
    config = write_config(
        tmp_path,
        {
            "traces": [{"bs": 32, "sqlite": "t.sqlite"}],
            "families": [{"label": "k", "symbol_patterns": ["^k$"]}],
        },
        sweep_json(
            avg_s=avg_s,
            decode_avg_s=decode_avg_s,
            prefill_avg_s=prefill_avg_s,
            tpot_s=decode_avg_s / 511,
            stored_share=phase_sum,
        ),
    )
    rc, mined, _ = run(tmp_path, config)
    assert rc == 0
    row = mined["per_bs"][0]
    assert row["decode_share_of_e2e"] == pytest.approx(decode_avg_s / avg_s)
    assert row["decode_share_of_e2e"] != pytest.approx(phase_sum)
    assert row["prefill_share"] == pytest.approx((avg_s - decode_avg_s) / avg_s)
    assert row["prefill_share"] != pytest.approx(prefill_avg_s / avg_s)
    assert row["stored_decode_share_of_e2e"] == pytest.approx(phase_sum)
    assert any("stored decode_share_of_e2e" in w for w in mined["warnings"])
    # No marker field is needed: the mismatch alone triggers the warning.
    assert "decode_share_of_phase_sum" not in mined["config_echo"]


# --- 4. residual / exhaustiveness -------------------------------------------


def test_non_exhaustive_partition_is_fatal_with_no_output(tmp_path):
    step = [(0, 1_000, "k"), (1_100, 2_000, "unmapped_kernel")]
    write_sqlite(tmp_path / "t.sqlite", transient_plus_step(step))
    config = write_config(
        tmp_path,
        {
            "traces": [{"bs": 32, "sqlite": "t.sqlite"}],
            "families": [{"label": "k", "symbol_patterns": ["^k$"]}],
        },
        sweep_json(decode_avg_s=1.9e-6 * 511, tpot_s=1.9e-6),
    )
    rc, mined, tables = run(tmp_path, config)
    assert rc == 1
    assert mined is None and tables is None


def test_allow_residual_lets_a_disclosed_remainder_through(tmp_path):
    step = [(0, 9_900, "k"), (10_000, 10_100, "unmapped_kernel")]
    write_sqlite(tmp_path / "t.sqlite", transient_plus_step(step))
    config = write_config(
        tmp_path,
        {
            "traces": [{"bs": 32, "sqlite": "t.sqlite"}],
            "families": [{"label": "k", "symbol_patterns": ["^k$"]}],
            "allow_residual": 2.0,
        },
        sweep_json(decode_avg_s=1.0e-5 * 511, tpot_s=1.0e-5),
    )
    rc, mined, _ = run(tmp_path, config)
    assert rc == 0
    row = mined["per_bs"][0]
    assert 0 < row["residual_pct"] <= 2.0
    assert row["partition_coverage"] < 1.0
    assert row["residual_symbols"] == ["unmapped_kernel"]
    assert any("__RESIDUAL__" in w for w in mined["warnings"])


# --- 5. decode_busy bounds --------------------------------------------------


def test_decode_busy_above_one_is_fatal(tmp_path):
    # 1 ms/step x 511 steps = 0.511 s of kernel work against a 0.2 s decode wall.
    step = [(0, 1_000_000, "k")]
    write_sqlite(tmp_path / "t.sqlite", transient_plus_step(step, gap_ns=2_000_000))
    config = write_config(
        tmp_path,
        {
            "traces": [{"bs": 32, "sqlite": "t.sqlite"}],
            "families": [{"label": "k", "symbol_patterns": ["^k$"]}],
        },
        sweep_json(avg_s=1.0, decode_avg_s=0.2, prefill_avg_s=0.05),
    )
    rc, mined, tables = run(tmp_path, config)
    assert rc == 1
    assert mined is None and tables is None


def test_decode_busy_below_floor_is_fatal(tmp_path):
    step = [(0, 1_000, "k")]
    write_sqlite(tmp_path / "t.sqlite", transient_plus_step(step))
    config = write_config(
        tmp_path,
        {
            "traces": [{"bs": 32, "sqlite": "t.sqlite"}],
            "families": [{"label": "k", "symbol_patterns": ["^k$"]}],
        },
        # 1 us/step x 511 = 0.000511 s against a 9 s decode wall → busy ~0.00006.
        sweep_json(avg_s=10.0, decode_avg_s=9.0, tpot_s=9.0 / 511),
    )
    rc, mined, tables = run(tmp_path, config)
    assert rc == 1
    assert mined is None and tables is None


def test_merged_busy_over_span_is_fatal():
    with pytest.raises(mine_trace.Fatal, match="exceeds window span"):
        mine_trace.check_window_integrity("synthetic", [(0, 100)], 200, 100)


# --- 6. step-count ladder + step_count_source ------------------------------


def _ladder_config(tmp_path, sweep):
    step = [(0, 1_000_000, "k")]
    write_sqlite(tmp_path / "t.sqlite", transient_plus_step(step, gap_ns=5_000_000))
    return write_config(
        tmp_path,
        {
            "traces": [{"bs": 32, "sqlite": "t.sqlite"}],
            "families": [{"label": "k", "symbol_patterns": ["^k$"]}],
        },
        sweep,
    )


def test_step_count_rung1_spec_decode_counters_win(tmp_path):
    sweep = sweep_json(
        avg_s=1.0,
        decode_avg_s=0.9,
        output_len=512,
        tpot_s=0.9 / 511,
        extra={"spec_decode": {"num_drafts": 256 * 10 * 32}},
    )
    rc, mined, _ = run(tmp_path, _ladder_config(tmp_path, sweep))
    assert rc == 0
    row = mined["per_bs"][0]
    assert row["step_count_source"] == "spec_decode.num_drafts"
    assert row["n_decode_steps"] == pytest.approx(256.0)
    assert set(row["step_count_candidates"]) == {
        "spec_decode.num_drafts",
        "decode_avg_s/tpot_s",
        "output_len-1",
    }


def test_step_count_rung2_tpot_is_used_when_no_spec_decode(tmp_path):
    sweep = sweep_json(avg_s=1.0, decode_avg_s=0.9, output_len=512, tpot_s=0.9 / 511)
    rc, mined, _ = run(tmp_path, _ladder_config(tmp_path, sweep))
    assert rc == 0
    row = mined["per_bs"][0]
    assert row["step_count_source"] == "decode_avg_s/tpot_s"
    assert row["n_decode_steps"] == pytest.approx(511.0)
    assert not any("tpot_s absent" in w for w in mined["warnings"])


def test_step_count_rung3_output_len_warns(tmp_path):
    sweep = sweep_json(avg_s=1.0, decode_avg_s=0.9, output_len=512)
    rc, mined, _ = run(tmp_path, _ladder_config(tmp_path, sweep))
    assert rc == 0
    row = mined["per_bs"][0]
    assert row["step_count_source"] == "output_len-1"
    assert row["n_decode_steps"] == pytest.approx(511.0)
    assert any("output_len-1" in w for w in mined["warnings"])


def test_step_count_rungs_disagreeing_is_fatal(tmp_path):
    # tpot implies ~256 steps; output_len-1 implies 511 → 100% apart.
    sweep = sweep_json(avg_s=1.0, decode_avg_s=0.9, output_len=512, tpot_s=0.9 / 256)
    rc, mined, tables = run(tmp_path, _ladder_config(tmp_path, sweep))
    assert rc == 1
    assert mined is None and tables is None


def test_step_count_unresolvable_is_fatal(tmp_path):
    sweep = sweep_json(avg_s=1.0, decode_avg_s=0.9, output_len=1)
    sweep["workload"].pop("output_len")
    rc, mined, tables = run(tmp_path, _ladder_config(tmp_path, sweep))
    assert rc == 1
    assert mined is None and tables is None


# --- 7. kernel-table probe + provenance -----------------------------------


@pytest.mark.parametrize(
    "table", ["CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"]
)
def test_kernel_table_probe_accepts_both_names(tmp_path, table):
    step = [(0, 1_000_000, "k")]
    write_sqlite(tmp_path / "t.sqlite", transient_plus_step(step, gap_ns=5_000_000), table=table)
    config = write_config(
        tmp_path,
        {
            "traces": [{"bs": 32, "sqlite": "t.sqlite"}],
            "families": [{"label": "k", "symbol_patterns": ["^k$"]}],
        },
        sweep_json(avg_s=1.0, decode_avg_s=0.9, tpot_s=0.9 / 511),
    )
    rc, mined, tables = run(tmp_path, config)
    assert rc == 0
    assert mined["provenance"]["kernel_table"] == table
    assert mined["provenance"]["nsys_product_version"] == "2026.3.1.157"
    assert mined["provenance"]["export_schema_version"] == "3.27.0"
    assert mined["schema"] == "mine_trace/1"
    assert tables.startswith(mine_trace.GENERATED_HEADER)
    assert "## Workload Dilution (per BS)" in tables
    assert "## Top Components (by f_e2e)" in tables


def test_missing_kernel_table_is_fatal(tmp_path):
    con = sqlite3.connect(str(tmp_path / "t.sqlite"))
    con.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    con.execute("CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT)")
    con.commit()
    con.close()
    config = write_config(
        tmp_path,
        {
            "traces": [{"bs": 32, "sqlite": "t.sqlite"}],
            "families": [{"label": "k", "symbol_patterns": ["^k$"]}],
        },
        sweep_json(tpot_s=0.9 / 511),
    )
    rc, mined, tables = run(tmp_path, config)
    assert rc == 1
    assert mined is None and tables is None


# --- 8. missing required column --------------------------------------------


def test_missing_required_column_is_fatal(tmp_path):
    # graphId dropped: in_graph_fraction would silently default to 0 without this.
    columns = (
        "start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, "
        "globalPid INTEGER, demangledName INTEGER"
    )
    step = [(0, 1_000_000, "k")]
    write_sqlite(
        tmp_path / "t.sqlite",
        transient_plus_step(step, gap_ns=5_000_000),
        columns=columns,
    )
    config = write_config(
        tmp_path,
        {
            "traces": [{"bs": 32, "sqlite": "t.sqlite"}],
            "families": [{"label": "k", "symbol_patterns": ["^k$"]}],
        },
        sweep_json(avg_s=1.0, decode_avg_s=0.9, tpot_s=0.9 / 511),
    )
    rc, mined, tables = run(tmp_path, config)
    assert rc == 1
    assert mined is None and tables is None


# --- 9. per-rank spread + in-graph fraction --------------------------------


def test_per_rank_spread_warns_and_in_graph_fraction_is_published(tmp_path):
    # rank A is gap-free; rank B idles half its span → spread ~0.5.
    step = [
        (0, 1_000_000, "k", 0, 111, 1),
        (0, 500_000, "k", 1, 222, None),
    ]
    write_sqlite(tmp_path / "t.sqlite", transient_plus_step(step, gap_ns=5_000_000))
    # widen rank B's span so its busy/span falls well below rank A's 1.0
    con = sqlite3.connect(str(tmp_path / "t.sqlite"))
    con.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end = start + 250000 WHERE deviceId = 1")
    con.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
        "(start, end, deviceId, streamId, globalPid, demangledName, graphId) "
        "SELECT start + 750000, end + 750000, 1, 7, 222, demangledName, NULL "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 1 LIMIT 1"
    )
    con.commit()
    con.close()
    config = write_config(
        tmp_path,
        {
            "traces": [{"bs": 32, "sqlite": "t.sqlite"}],
            "families": [{"label": "k", "symbol_patterns": ["^k$"]}],
        },
        sweep_json(avg_s=1.0, decode_avg_s=0.9, tpot_s=0.9 / 511),
    )
    rc, mined, _ = run(tmp_path, config)
    assert rc == 0
    depth = mined["per_bs"][0]["depths"][0]
    assert len(depth["per_rank"]) == 2
    assert depth["per_rank_busy_spread"] > mine_trace.MAX_RANK_BUSY_SPREAD
    assert any("per-rank busy spread" in w for w in mined["warnings"])
    assert 0.0 < depth["in_graph_fraction"] < 1.0


# --- 10. stable mined.json subset other tools may read ---------------------


def test_stable_subset_and_f_e2e_are_structurally_consistent(tmp_path):
    step = [
        (0, 700_000, "big_gemm"),
        (700_100, 1_000_000, "small_norm"),
    ]
    write_sqlite(tmp_path / "t.sqlite", transient_plus_step(step, gap_ns=5_000_000))
    config = write_config(
        tmp_path,
        {
            "traces": [{"bs": 32, "sqlite": "t.sqlite", "label": "step 512"}],
            "families": [
                {"label": "gemm", "symbol_patterns": ["big_gemm"], "physical_ceiling": 2.0,
                 "prefill_active": True},
                {"label": "norm", "symbol_patterns": ["small_norm"], "prefill_active": False},
            ],
        },
        sweep_json(avg_s=1.0, decode_avg_s=0.9, prefill_avg_s=0.05, tpot_s=0.9 / 511),
    )
    rc, mined, tables = run(tmp_path, config)
    assert rc == 0
    row = mined["per_bs"][0]
    for key in (
        "bs",
        "trace",
        "decode_avg_s",
        "avg_s",
        "decode_share_of_e2e",
        "decode_busy",
        "n_decode_steps",
        "step_count_source",
        "in_graph_fraction",
        "families",
        "residual_pct",
        "partition_coverage",
    ):
        assert key in row, key
    assert row["decode_share_of_e2e"] == pytest.approx(row["decode_avg_s"] / row["avg_s"])
    assert sum(f["f_decode"] for f in row["families"]) == pytest.approx(1.0)
    for fam in row["families"]:
        assert fam["f_e2e"] == pytest.approx(
            fam["f_decode"] * row["decode_busy"] * row["decode_share_of_e2e"]
        )
    assert row["families"][0]["label"] == "gemm"
    assert row["families"][0]["addressable_e2e"] == pytest.approx(
        row["families"][0]["f_e2e"] * 0.5
    )
    assert "2.000x" in tables and "(disclosed)" in tables


# --- 11. generated tables.md stays machine-readable for the mining enrichment -


def test_generated_tables_parse_through_ammo_state_enrichment(tmp_path):
    """ammo_state.parse_mining_md feeds the Stage-7 stop decision from this table."""
    state_script = Path(__file__).resolve().parents[1] / "scripts" / "ammo_state.py"
    spec = importlib.util.spec_from_file_location("ammo_state_for_mine", str(state_script))
    ammo_state = importlib.util.module_from_spec(spec)
    sys.modules["ammo_state_for_mine"] = ammo_state
    spec.loader.exec_module(ammo_state)

    # 1.7613 ms/step x 511 ~= the 0.9 s decode wall, so decode_busy ~1.0 and the
    # gemm family dominates — the realistic shape of a decode-bound campaign.
    step = [(0, 1_232_910, "big_gemm"), (1_233_010, 1_761_300, "small_norm")]
    write_sqlite(tmp_path / "t.sqlite", transient_plus_step(step, gap_ns=5_000_000))
    config = write_config(
        tmp_path,
        {
            "traces": [{"bs": 32, "sqlite": "t.sqlite"}],
            "families": [
                {"label": "gemm", "symbol_patterns": ["big_gemm"], "physical_ceiling": 2.0},
                {"label": "norm", "symbol_patterns": ["small_norm"]},
            ],
        },
        sweep_json(avg_s=1.0, decode_avg_s=0.9, prefill_avg_s=0.05, tpot_s=0.9 / 511),
    )
    rc, mined, tables = run(tmp_path, config)
    assert rc == 0
    parsed = ammo_state.parse_mining_md(tables)
    row = mined["per_bs"][0]
    assert parsed["top_component"] == "gemm"
    assert parsed["top_f_e2e_pct"] == pytest.approx(round(row["families"][0]["f_e2e"], 4) * 100)
    assert parsed["top_addressable_e2e_pct"] == pytest.approx(
        round(row["families"][0]["addressable_e2e"], 4) * 100
    )
    assert parsed["decode_frac"] == pytest.approx(round(row["decode_share_of_e2e"], 4))


# --- 12. the two variant copies stay byte-identical -------------------------


def test_variant_copies_are_byte_identical():
    """mine_trace.py is pure computation: it must never fork per variant."""
    session_root = Path(__file__).resolve().parents[4]
    copies = [
        session_root / variant / "skills" / "ammo" / "scripts" / "mine_trace.py"
        for variant in (".claude", ".codex")
    ]
    present = [c for c in copies if c.is_file()]
    if len(present) < 2:
        pytest.skip("only one variant is materialized in this tree")
    assert present[0].read_bytes() == present[1].read_bytes()
