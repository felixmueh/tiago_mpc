#!/usr/bin/env python3
"""Apply right-arm dynamics metadata without changing model geometry/inertia.

The canonical source is preserved byte for byte. The retained reduced arm was
already extracted from it with the gripper lumped at its neutral posture; this
script patches its metadata in place, without repeating that reduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SOURCE = MODELS / "tiago_pro/tiago_pro_sim_source.urdf"
FULL = MODELS / "tiago_pro/tiago_pro.urdf"
ARM = MODELS / "tiago_right/tiago_right_arm.urdf"
COULOMB_NM = (2.1, 2.6, 2.0, 1.95, 0.8, 1.1, 0.3)
VISCOUS_NM_S_PER_RAD = (3.35, 4.15, 2.45, 2.25, 0.5, 0.3, 0.15)
NOTES = (
    "User c estimate; b midpoint of 1.8-4.9 (2 runs)",
    "User c/b midpoints of 2.3-2.9 / 2.7-5.6 (8 runs)",
    "User c/b midpoints of 1.8-2.2 / 1.3-3.6 (8 runs)",
    "User c/b midpoints of 1.8-2.1 / 1.1-3.4 (8 runs)",
    "ASSUMED c=0.8 above user lower bound >0.7; ASSUMED b=0.5, unmeasured",
    "User c approximately 1.1 (1 run); ASSUMED b=0.3, unmeasured",
    "ASSUMED c=0.3 and b=0.15, both unmeasured",
)


def patch_dynamics(text):
    """Replace seven dynamics tags/comments; preserve every other source byte."""
    for i, (c, b, note) in enumerate(zip(COULOMB_NM, VISCOUS_NM_S_PER_RAD, NOTES), 1):
        pattern = rf'(<joint\b(?=[^>]*\btype="revolute")[^>]*\bname="arm_right_{i}_joint"[^>]*>)(.*?)(</joint>)'

        def patch_joint(match):
            body = re.sub(r'(?m)^[ \t]*<!-- TIAGO_MPC dynamics:.*?-->\r?\n', '', match[2])
            tag = re.compile(r'(?m)^([ \t]*)<dynamics\b[^>]*/>[ \t]*$')

            def replace_dynamics(dynamics):
                indent = dynamics[1]
                comment = (f'{indent}<!-- TIAGO_MPC dynamics: {note}; c in Nm, '
                           'b in Nm s/rad. Metadata only; not a calibrated stiction model. -->')
                return comment + '\n' + f'{indent}<dynamics damping="{b}" friction="{c}" />'

            body, count = tag.subn(replace_dynamics, body)
            if count != 1:
                raise ValueError(f"Expected one dynamics tag for arm_right_{i}_joint, got {count}")
            return match[1] + body + match[3]

        text, count = re.subn(pattern, patch_joint, text, flags=re.DOTALL)
        if count != 1:
            raise ValueError(f"Expected one arm_right_{i}_joint, got {count}")
    return text


def asset_manifest(full, arm):
    paths = [SOURCE, FULL, ARM, *sorted((MODELS / "tiago_pro/meshes").glob("*.stl"))]
    replacements = {FULL: full.encode(), ARM: arm.encode()}
    return {str(p.relative_to(MODELS)): dict(
        sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
        for p in paths for payload in [replacements.get(p, p.read_bytes())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify prepared models/manifest without writing")
    args = parser.parse_args()
    full = patch_dynamics(SOURCE.read_text())
    arm = patch_dynamics(ARM.read_text())
    manifest = json.dumps(asset_manifest(full, arm), indent=2, sort_keys=True) + '\n'
    outputs = {FULL: full, ARM: arm, MODELS / "manifest.json": manifest}
    for path, expected in outputs.items():
        if args.check:
            if path.read_bytes() != expected.encode():
                raise ValueError(f"Run scripts/prepare_models.py to update {path}")
        else:
            path.write_text(expected)
    print("Prepared model metadata and manifest are current" if args.check else
          "Prepared full model, patched retained reduced arm metadata and updated manifest")


if __name__ == "__main__":
    main()
