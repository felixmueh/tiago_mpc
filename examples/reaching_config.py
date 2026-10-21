"""Compatibility imports for existing experiment scripts; implementation is packaged."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tiago_mpc.configuration import (DEFAULT_CONFIG, LEGACY_PARAMS, LEGACY_N,
                                     LEGACY_DT_S, load_configuration, resolve_configuration)
