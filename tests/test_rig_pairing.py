"""Unit checks for dual-lens pairing helpers (no COLMAP required)."""

from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from splatking.rig import (  # noqa: E402
    frame_basename,
    pair_by_frame_times,
    select_pairs_keep_pct,
    write_cross_lens_pairs_file,
    write_rig_config,
)


def test_pair_by_frame_times():
    wide = [0.0, 0.033, 0.066, 0.100]
    ultra = [0.00008, 0.0331, 0.0662, 0.0999]  # ~0.08–0.2 ms drift
    pairs = pair_by_frame_times(wide, ultra, tolerance_s=0.005)
    assert len(pairs) == 4
    assert pairs[0].wide_idx == 0 and pairs[0].ultra_idx == 0
    assert pairs[0].dt_s < 0.001


def test_pair_tolerance_boundary():
    wide = [0.0, 1.0]
    ultra = [0.006, 1.0]  # 6ms > 5ms tol → first unpaired
    pairs = pair_by_frame_times(wide, ultra, tolerance_s=0.005)
    assert len(pairs) == 1
    assert pairs[0].wide_idx == 1


def test_select_keep_pct_and_names():
    wide = [i * 0.033 for i in range(100)]
    ultra = [t + 0.0001 for t in wide]
    pairs = pair_by_frame_times(wide, ultra)
    kept = select_pairs_keep_pct(pairs, 0.10)
    assert 8 <= len(kept) <= 12
    assert frame_basename(kept[0].pair_id).startswith("frame_")


def test_write_files():
    with tempfile.TemporaryDirectory() as td:
        pairs = write_cross_lens_pairs_file(
            os.path.join(td, "pairs.txt"),
            ["frame_000001.jpg", "frame_000002.jpg"],
        )
        assert pairs == 2
        text = open(os.path.join(td, "pairs.txt"), encoding="utf-8").read()
        assert "wide/frame_000001.jpg ultra/frame_000001.jpg" in text
        cfg = os.path.join(td, "rig_config.json")
        write_rig_config(cfg)
        assert os.path.isfile(cfg)


if __name__ == "__main__":
    test_pair_by_frame_times()
    test_pair_tolerance_boundary()
    test_select_keep_pct_and_names()
    test_write_files()
    print("ok")
