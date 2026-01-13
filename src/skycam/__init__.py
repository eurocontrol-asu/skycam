"""skycam - Fisheye image projection for ground-based sky observation."""

# Re-export subpackages for convenience
from skycam import adapters, config, domain
from skycam._version import __version__

__all__ = [
    "__version__",
    "adapters",
    "config",
    "domain",
]
