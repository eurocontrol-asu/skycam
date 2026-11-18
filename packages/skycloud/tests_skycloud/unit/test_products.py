from __future__ import annotations

import numpy as np

from skycloud.products import value_code


def test_value_code() -> None:
    code = value_code(np.array([0.0, 0.5, 1.0, np.nan], np.float32))
    assert code.tolist() == [0, 127, 254, 255]
