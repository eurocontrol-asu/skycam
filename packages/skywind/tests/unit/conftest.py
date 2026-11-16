from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def era5_ds() -> xr.Dataset:
    """Small synthetic ERA5-like dataset (time, level, latitude, longitude) with uniform winds u=10, v=-5 m/s."""
    time = pd.date_range("2024-03-18T12:00", periods=4, freq="1h").values
    level = np.array([150.0, 200.0, 250.0, 300.0, 350.0, 400.0])
    lat = np.arange(46.0, 51.01, 0.25)
    lon = np.arange(0.0, 5.01, 0.25)
    shape = (len(time), len(level), len(lat), len(lon))
    data = {
        "u": np.full(shape, 10.0),
        "v": np.full(shape, -5.0),
        "t": np.full(shape, 220.0),
        "q": np.full(shape, 1e-5),
        "w": np.zeros(shape),
        "z": np.full(shape, 1e5),
    }
    return xr.Dataset(
        {k: (("time", "level", "latitude", "longitude"), v) for k, v in data.items()},
        coords={"time": time, "level": level, "latitude": lat, "longitude": lon},
    )
