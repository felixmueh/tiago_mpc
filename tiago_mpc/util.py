"""Shared infrastructure: the cache that GATO's modules and every derived robot file
share, with content-addressed writes.

cache_root exports GATO_CACHE_DIR because GATO finds its cache only through that variable
(without it, a per-user directory): GATO's modules then land beside the derived URDFs
they are compiled from, and spawned processes (MpcProcess, the safety worker) inherit
the same cache.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Callable
import xml.etree.ElementTree as ET


def main_checkout(repo: Path) -> Path:
    """The main working tree of repo's git repository (repo itself without git), so
    every worktree shares the main checkout's cache."""
    try:
        common = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return repo
    return Path(common).parent if Path(common).name == ".git" else repo


def cache_root() -> Path:
    """GATO_CACHE_DIR if set, else the main checkout's .cache/gato, shared by every
    worktree, host and container (Felix, 2026-10-23); exported for GATO."""
    default = main_checkout(Path(__file__).resolve().parents[1]) / ".cache/gato"
    root = Path(os.environ.get("GATO_CACHE_DIR") or default).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["GATO_CACHE_DIR"] = str(root)
    return root


def cached(name: str, payload: bytes) -> Path:
    """payload as <cache_root()>/models/<hash of payload>/name. Content-addressed, so
    concurrent processes deriving different files never share a path."""
    return _write(cache_root() / "models" / digest(payload) / name, payload)


def _write(path: Path, payload: bytes) -> Path:
    """payload at path, atomically, and only when missing or different."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path.exists() and path.read_bytes() == payload):
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    return path


# The derived URDFs reference cached copies of their meshes, named by their contents
# (models/meshes/<sha256>.stl), by paths relative to the URDF (models/<digest>/<name>.urdf).
# GATO spherizes the GPU copy's collision meshes into its module, but its codegen key
# hashes only the URDF's bytes: a content name makes a changed mesh change those bytes,
# and a relative path keeps them, and the plant names, independent of the checkout's path.
def cache_meshes(tree: ET.Element, source: Callable[[str], Path]) -> None:
    """Point every <mesh> in tree at a cached copy of source(its filename)."""
    for mesh in tree.iter("mesh"):
        path = source(mesh.attrib["filename"])
        data = path.read_bytes()
        if data.startswith(b"version https://git-lfs"):
            raise ValueError(f"Git LFS pointer instead of a mesh (git lfs pull): {path}")
        name = hashlib.sha256(data).hexdigest() + path.suffix.lower()
        _write(cache_root() / "models/meshes" / name, data)
        mesh.set("filename", f"../meshes/{name}")


def digest(*parts: bytes) -> str:
    """Eight hex characters over the parts (length-prefixed, so order and splits matter)."""
    h = hashlib.sha256()
    for part in parts:
        h.update(len(part).to_bytes(8, "little") + part)
    return h.hexdigest()[:8]
