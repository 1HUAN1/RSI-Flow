"""SIA: Self-Improving AI framework"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sia-agent")
except PackageNotFoundError:  # package is not installed (e.g. running from source)
    __version__ = "0.0.0+unknown"
