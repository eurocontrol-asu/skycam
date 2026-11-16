from __future__ import annotations

from skywind.sources.remote import RemoteCopySource


def test_remote_path_pattern() -> None:
    src = RemoteCopySource("host", "/archive/{Y}/{Y}_{m}_{d}.pl.nc")
    assert src.remote_path("20240318") == "/archive/2024/2024_03_18.pl.nc"
