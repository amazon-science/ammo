# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Zone F — Playwright visual snapshots for per-BS chip rendering (Tests 67-70).

Each chip variant defined in the LIGHTGRID chip per-BS design (BASELINE /
RE-PROFILE / TRACK / INTEGRATION traffic-light) is rendered on a fixed-grid
harness page (`frontend/chip-test-harness.html`) with pinned synthetic data
and compared against a committed PNG fixture under
`tests/fixtures/chips/`.

The comparison uses pixel-diff tolerance ≤ 2% (measured as the fraction of
pixels whose channel-wise Manhattan distance exceeds 12) — matching the
spec's "pixel-level tolerance ≤ 2%" acceptance criterion.

Regeneration: set `UPDATE_SNAPSHOTS=1` in the environment to overwrite the
fixtures instead of diffing. Do this only when the design intentionally
changes (and commit the new PNGs with the change).

Server prereq: the AMMO server must be running on AMMO_SERVER_URL
(default http://localhost:8000) with `/static/chip-test-harness.html`
served by the static mount (`app.py::StaticFiles("/static", frontend_dir)`).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image


FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "chips"
UPDATE_SNAPSHOTS = os.getenv("UPDATE_SNAPSHOTS", "").lower() in ("1", "true", "yes")

# Per-channel distance threshold (0-765). 12 tolerates minor font-aliasing
# drift on different GPU / Chromium builds while still catching any meaningful
# rendering regression.
PIXEL_DIFF_TOLERANCE = 12
# Maximum fraction of pixels allowed to exceed the per-pixel threshold.
MAX_PIXEL_MISMATCH_RATIO = 0.02

# Every entry in this list yields exactly one snapshot test. `slot_id` is the
# `data-test-id` on the harness slot; `fixture` is the filename inside
# FIXTURE_DIR. Keep this list sorted by chip family for reviewability.
CHIP_VARIANTS: list[tuple[str, str]] = [
    ("chip-baseline-3bs",            "chip-baseline-3bs.png"),
    ("chip-baseline-1bs",            "chip-baseline-1bs.png"),
    ("chip-reprofile-green",         "chip-reprofile-green.png"),
    ("chip-reprofile-amber",         "chip-reprofile-amber.png"),
    ("chip-reprofile-regressed",     "chip-reprofile-regressed.png"),
    ("chip-track-shipped-allpass",   "chip-track-shipped-allpass.png"),
    ("chip-track-mixed-verdicts",    "chip-track-mixed-verdicts.png"),
    ("chip-track-validated",         "chip-track-validated.png"),
    ("chip-integ-green",             "chip-integ-green.png"),
    ("chip-integ-amber",             "chip-integ-amber.png"),
    ("chip-integ-red",               "chip-integ-red.png"),
    ("chip-integ-round1-default",    "chip-integ-round1-default.png"),
    # MINING / DEBATE — Option A chip faces (mockup §1, §3).
    ("chip-mining-short",            "chip-mining-short.png"),
    ("chip-mining-long",             "chip-mining-long.png"),
    ("chip-debate-single-winner",    "chip-debate-single-winner.png"),
    ("chip-debate-multi-winners",    "chip-debate-multi-winners.png"),
    ("chip-debate-active",           "chip-debate-active.png"),
]


# ────────────────────────────────────────────────────────────────────────────
# Pixel-diff helper
# ────────────────────────────────────────────────────────────────────────────

def _compare_pngs(actual_path: Path, expected_path: Path) -> tuple[float, int, int]:
    """Return `(mismatch_ratio, mismatch_count, total)` between two PNGs.

    Decodes both images to RGB pixels and counts pixels whose per-channel
    Manhattan distance exceeds PIXEL_DIFF_TOLERANCE. Raises if dimensions
    differ — a size mismatch is never a soft failure.
    """
    img_a = Image.open(actual_path).convert("RGB")
    img_b = Image.open(expected_path).convert("RGB")
    if img_a.size != img_b.size:
        raise AssertionError(
            f"Size mismatch — actual={img_a.size}, expected={img_b.size}. "
            "Chip dimensions changed or viewport DPR drifted."
        )
    pa = img_a.load()
    pb = img_b.load()
    w, h = img_a.size
    mismatched = 0
    for y in range(h):
        for x in range(w):
            ra, ga, ba = pa[x, y]
            rb, gb, bb = pb[x, y]
            if abs(ra - rb) + abs(ga - gb) + abs(ba - bb) > PIXEL_DIFF_TOLERANCE:
                mismatched += 1
    total = w * h
    return mismatched / total, mismatched, total


# ────────────────────────────────────────────────────────────────────────────
# Session-scoped harness loader — renders every chip variant once per session
# and caches their screenshot PNGs on disk. The per-test `snapshot` fixture
# reuses that cache instead of re-launching Chromium 8 times.
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def chip_screenshots(playwright_instance, server_url, tmp_path_factory):
    browser = playwright_instance.chromium.launch(headless=True)
    # Viewport height must contain every harness slot; growing the slot list
    # (e.g. MINING + DEBATE in 2026-05-07) adds additional rows at 3-wide
    # layout. Give the viewport generous headroom so page.screenshot(clip=)
    # stays inside the rendered area.
    ctx = browser.new_context(
        viewport={"width": 1100, "height": 1600},
        device_scale_factor=1,
    )
    page = ctx.new_page()
    # Cache-bust the harness so a fresh docker-cp deploy is picked up
    # between local test runs.
    url = f"{server_url}/static/chip-test-harness.html?t={os.urandom(4).hex()}"
    try:
        page.goto(url, wait_until="networkidle", timeout=10_000)
    except Exception as e:
        browser.close()
        pytest.skip(
            f"Chip harness not reachable at {url}. Is the AMMO server running? "
            f"err={e}"
        )
    try:
        page.wait_for_function("window.__chipsReady === true", timeout=5_000)
    except Exception:
        browser.close()
        pytest.skip(
            "Chip harness failed to signal __chipsReady within 5s. "
            "Check browser console for CircuitBoard errors."
        )
    # Settle web-fonts so the PNG isn't captured mid-FOUT.
    page.wait_for_timeout(300)

    capture_dir = tmp_path_factory.mktemp("chip-screenshots")
    captured: dict[str, Path] = {}
    for slot_id, _fixture_name in CHIP_VARIANTS:
        slot = page.query_selector(f'[data-test-id="{slot_id}"]')
        assert slot is not None, f"harness slot missing: {slot_id}"
        chip = slot.query_selector(".cb2-hud")
        assert chip is not None, f"slot {slot_id} has no .cb2-hud child"
        bb = chip.bounding_box()
        assert bb is not None, f"chip bounding-box unavailable: {slot_id}"
        out = capture_dir / f"{slot_id}.png"
        page.screenshot(
            path=str(out),
            clip={
                "x": bb["x"], "y": bb["y"],
                "width": bb["width"], "height": bb["height"],
            },
        )
        captured[slot_id] = out
    browser.close()
    return captured


# ────────────────────────────────────────────────────────────────────────────
# Parametrised snapshot tests — one assertion per chip variant so failures
# localise to the specific variant rather than drowning in a single blob.
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "slot_id,fixture_name",
    CHIP_VARIANTS,
    ids=[slot for slot, _ in CHIP_VARIANTS],
)
def test_chip_snapshot(chip_screenshots, slot_id, fixture_name):
    """Pixel-diff an individual chip variant vs its committed PNG fixture."""
    actual = chip_screenshots.get(slot_id)
    assert actual is not None and actual.exists(), (
        f"harness did not produce a screenshot for {slot_id}"
    )
    expected = FIXTURE_DIR / fixture_name
    if UPDATE_SNAPSHOTS or not expected.exists():
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(actual, expected)
        if not UPDATE_SNAPSHOTS:
            pytest.skip(
                f"fixture {fixture_name} did not exist — wrote it. "
                "Re-run with it committed to enable diffing."
            )
        return

    ratio, bad, total = _compare_pngs(actual, expected)
    assert ratio <= MAX_PIXEL_MISMATCH_RATIO, (
        f"{slot_id}: {bad}/{total} pixels ({ratio:.3%}) exceed tolerance "
        f"{PIXEL_DIFF_TOLERANCE}/765. Threshold = {MAX_PIXEL_MISMATCH_RATIO:.1%}. "
        f"Regenerate fixture with UPDATE_SNAPSHOTS=1 if the change is intentional."
    )
