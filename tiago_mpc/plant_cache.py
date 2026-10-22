"""Plant names from model identity and the one GATO cache every caller shares.

GATO compiles one CUDA module per registered plant name and N (minutes). A fixed
name per variant would let two variants overwrite each other in a shared cache,
so the name hashes everything compiled into the module. The hash is a cache key,
not provenance.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(*parts: bytes) -> str:
    """Eight hex characters over the parts (length-prefixed, so order and splits matter)."""
    h = hashlib.sha256()
    for part in parts:
        h.update(len(part).to_bytes(8, "little") + part)
    return h.hexdigest()[:8]


def cache_root() -> Path:
    """GATO_CACHE_DIR if set (host runs), else <repo>/.cache/gato; exported for GATO."""
    root = Path(os.environ.get("GATO_CACHE_DIR") or ROOT / ".cache/gato").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["GATO_CACHE_DIR"] = str(root)
    return root


def plant_name(gpu_urdf_path, ee_frame: str, losses=None) -> str:
    """tiago_right_<hash> over the GPU URDF bytes, end-effector frame and predictor
    friction configuration as passed to registration (coefficients and width)."""
    config = json.dumps(losses and {k: v for k, v in losses.items() if k != "provenance"},
                        sort_keys=True).encode()
    return "tiago_right_" + digest(Path(gpu_urdf_path).read_bytes(), ee_frame.encode(), config)
