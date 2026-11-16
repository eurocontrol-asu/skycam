"""GVCCS annotations and Flightradar24 tracks, train split only (the test split is never opened)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["GvccsDay", "load_day", "train_days"]


def _train(root: Path) -> Path:
    path = root / "train"
    if path.name != "train":  # pragma: no cover - defensive
        raise ValueError("only the GVCCS train split may be read")
    return path


@lru_cache(maxsize=2)
def _annotations(root: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((_train(root) / "annotations.json").read_text())
    return data


def train_days(root: Path) -> list[str]:
    return sorted({i["file_name"][6:14] for i in _annotations(root)["images"]})


@dataclass(frozen=True)
class GvccsDay:
    """Frames of a day (``image_id`` -> UTC time) and the polygons of flight-attributed contrails."""

    day: str
    frames: pd.DataFrame  # image_id, time, file_name
    polygons: pd.DataFrame  # image_id, time, contrail_id, flight (GVCCS flight id), polygon (list of x, y)
    tracks: pd.DataFrame  # Flightradar24: FLIGHT_ID, CALL_SIGN, REGISTRATION, ICAO_TYPE, timestamp, LATITUDE, ...


def load_day(root: Path, day: str) -> GvccsDay:
    ann = _annotations(root)
    imgs = [i for i in ann["images"] if i["file_name"][6:14] == day]
    frames = pd.DataFrame(
        {
            "image_id": [i["id"] for i in imgs],
            "file_name": [i["file_name"] for i in imgs],
            "time": pd.to_datetime([i["file_name"][6:20] for i in imgs], format="%Y%m%d%H%M%S", utc=True),
        }
    )
    frames["time"] = frames["time"].dt.as_unit("ns")
    when = dict(zip(frames["image_id"], frames["time"], strict=True))
    rows = []
    for a in ann["annotations"]:
        if a["image_id"] in when and a.get("flight_id"):
            seg = a["segmentation"]
            seg = json.loads(seg) if isinstance(seg, str) else seg
            for poly in seg:
                if len(poly) >= 6:
                    rows.append(
                        {
                            "image_id": a["image_id"],
                            "time": when[a["image_id"]],
                            "contrail_id": a["contrail_id"],
                            "flight": str(a["flight_id"]),
                            "polygon": poly,
                        }
                    )
    tracks = pd.read_parquet(next((_train(root) / "parquet").glob(f"{day}*.parquet")))
    tracks["timestamp"] = pd.to_datetime(tracks["TIMESTAMP_S"]).dt.tz_localize("UTC").dt.as_unit("ns")
    return GvccsDay(day, frames.sort_values("time").reset_index(drop=True), pd.DataFrame(rows), tracks)
