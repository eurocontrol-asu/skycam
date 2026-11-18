from __future__ import annotations

import json

import numpy as np

from skycloud.ledger import DayLedger, SmallFrame, offset_from, step_delta

PRM = {"T_STEP": 0.1, "R_STEP": 0.03, "N_NOISE": 0.006, "SIG_D": 2.0}


def _frame(shift: float) -> SmallFrame:
    sky = np.ones((40, 40), bool)
    return SmallFrame(
        10.0, np.full((40, 40), 0.1 + shift, np.float32), np.ones((40, 40), np.float32), sky
    )


def test_step_delta_measures_the_offset_on_unchanged_sky() -> None:
    d, sup, var = step_delta(_frame(0.0), _frame(0.015), PRM, c10=True)
    assert np.allclose(d, 0.015, atol=1e-5) and sup == 1.0
    assert var is not None and var.max() < 1e-4
    d5, _, v5 = step_delta(_frame(0.0), _frame(0.015), PRM, c10=False)
    assert v5 is None and np.allclose(d5, 0.015, atol=1e-5)


def _ledger(var: bool) -> DayLedger:
    frames = np.array(
        [f"202401010{m // 60 + 6}{m % 60:02d}{s:02d}" for m in range(30, 90) for s in (0, 30)]
    )
    delta = np.full((1, 4, 4), 0.02, np.float32)
    return DayLedger(
        t0=np.array(["20240101060000"]), t1=np.array(["20240101070000"]), delta=delta,
        support=np.array([1.0]),
        var=np.full((1, 4, 4), 1e-8, np.float32) if var else np.zeros((0, 4, 4), np.float32),
        frames=frames, ped=np.ones(len(frames)),
        anchors=np.array(["20240101080000", "20240101150000"]),
        params=json.dumps(PRM),
    )  # fmt: skip


def test_offset_removes_a_morning_step_as_a_discontinuity() -> None:
    sky = np.ones((4, 4), bool)
    before, note = offset_from(_ledger(False), "20240101065930", sky, 8.0)
    after, _ = offset_from(_ledger(False), "20240101070000", sky, 8.0)
    assert before is not None and after is not None
    jump = float(before[0, 0] - after[0, 0])  # the step is cancelled across it, the average is not
    assert 0.018 < jump <= 0.02 and before[0, 0] > 0 > after[0, 0]
    assert "1 step(s)" in note
    noisy, note2 = offset_from(_ledger(True), "20240101065930", sky, 8.0)
    assert noisy is not None and "noise check" in note2


def test_offset_without_ledger_or_frame() -> None:
    sky = np.ones((4, 4), bool)
    assert offset_from(None, "20240101060000", sky, 8.0)[0] is None
    assert offset_from(_ledger(False), "20240101063015", sky, 8.0)[0] is None
