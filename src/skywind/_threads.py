"""Cap BLAS / OpenMP threads (the host is shared and heavily loaded) unless the caller set them explicitly.

Imported by the command-line entry point before numpy; library users control threads themselves.
"""

from __future__ import annotations

import os

__all__ = ["DEFAULT_THREADS"]

DEFAULT_THREADS = "4"
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, DEFAULT_THREADS)
