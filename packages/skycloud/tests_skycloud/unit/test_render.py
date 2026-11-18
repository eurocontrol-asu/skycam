from __future__ import annotations

from skycloud.params import CloudMapParams
from skycloud.render import ambiguity
from skycloud.tables import ReferenceTables

P = CloudMapParams()


def _row(ts: str, **kw: object) -> dict[str, object]:
    return {"ts": ts, "elev": 4.0, **kw}


def test_contradiction_and_deck_weights_are_smoothed_in_time(tiny_tables: ReferenceTables) -> None:
    rows = [
        _row("20240101070000", lift={"w_ov": 1.0, "w_col": 0.0}),
        _row("20240101070030", lift={"w_ov": 1.0, "w_col": 1.0}, overcast_level={"q5_s": 2.0}),
        _row("20240101120000", elev=20.0, lift={"w_ov": 1.0, "w_col": 0.0}),  # daytime: not read
    ]
    out = ambiguity(rows, tiny_tables, P)
    a, b = out["20240101070000"], out["20240101070030"]
    assert 0.5 < a["C"] < 1.0 and 0.0 < a["W"] < 0.5
    assert b["Wg"] == b["W"]  # q5_s above the overcast p5: no clear-coloured gap
    assert "20240101120000" not in out


def test_ambiguity_of_a_shape_ambiguous_structureless_frame(tiny_tables: ReferenceTables) -> None:
    rows = [_row("20240101070000", joint_s=0.5, struct_share=0.0)]
    out = ambiguity(rows, tiny_tables, P)["20240101070000"]
    assert out["a_struct"] == 1.0 and 0.7 < out["a_shape"] <= 1.0
    assert out["A"] == round(out["a_shape"] * 1.0 * 1.0, 4)
    clear = ambiguity([_row("20240101070000", joint_s=0.95, struct_share=0.0)], tiny_tables, P)
    assert clear["20240101070000"]["A"] == 0.0
