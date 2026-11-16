"""Keep research-store tests opt-in, including on configured local machines."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip research-store cases unless the operator explicitly enables them."""
    if os.environ.get("SKYWIND_TEST_STORES") == "1":
        return
    skip = pytest.mark.skip(reason="Set SKYWIND_TEST_STORES=1 to use research stores")
    for item in items:
        if item.get_closest_marker("requires_store") is not None:
            item.add_marker(skip)
