#!/usr/bin/env python3
"""Apply supplied arm joint parameters without changing model geometry/inertia.

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
# Felix's actual parameter table, 2026-10-21. No kp/kd: those belong to a
# different controller. Columns: Coulomb Nm, viscous Nm s/rad, armature kg m^2,
# actuator effort Nm. Armature is metadata only; no runtime currently reads it.
PARAMETERS = {
    "left": (
        (2.47, 7.42, .511, 43), (3.52, 8.71, 1.19, 43),
        (1.04, 4.75, .472, 26), (1.35, 3.65, .296, 26),
        (.779, 3.65, .424, 26), (.773, 2.82, .151, 26),
        (.945, 1.79, .14, 26),
    ),
    "right": (
        (1.66, 4.7, .746, 43), (1.15, 9.69, .882, 43),
        (1.16, 5.3, .211, 26), (.912, 4.2, .648, 26),
        (1.25, 3.66, .326, 26), (.473, 2.95, .343, 26),
        (.56, 2.95, .27, 26),
    ),
}


def patch_dynamics(text, sides=("left", "right")):
    """Patch only arm dynamics, effort and custom armature metadata."""
    for side in sides:
        for i, (c, b, armature, effort) in enumerate(PARAMETERS[side], 1):
            name = f"arm_{side}_{i}_joint"
            pattern = rf'(<joint\b(?=[^>]*\btype="revolute")[^>]*\bname="{name}"[^>]*>)(.*?)(</joint>)'

            def patch_joint(match):
                body = re.sub(r'(?m)^[ \t]*<!-- TIAGO_MPC dynamics:.*?-->\r?\n', '', match[2])
                body = re.sub(r'(?m)^[ \t]*<tiago_mpc\b[^>]*/>[ \t]*\r?\n', '', body)
                tag = re.compile(r'(?m)^([ \t]*)<dynamics\b[^>]*/>[ \t]*$')

                def replace_dynamics(dynamics):
                    indent = dynamics[1]
                    comment = (f'{indent}<!-- TIAGO_MPC dynamics: Actual values supplied by Felix '
                               '2026-10-21; friction Nm, damping Nm s/rad; armature kg m^2 '
                               'is custom metadata, not enabled by URDF loaders. -->')
                    return (comment + '\n' + f'{indent}<dynamics damping="{b}" friction="{c}" />'
                            + '\n' + f'{indent}<tiago_mpc armature="{armature}" />')

                body, count = tag.subn(replace_dynamics, body)
                if count != 1:
                    raise ValueError(f"Expected one dynamics tag for {name}, got {count}")
                body, count = re.subn(r'(<limit\b[^>]*\beffort=")[^"]*(")',
                                      lambda m: m[1] + str(effort) + m[2], body)
                if count != 1:
                    raise ValueError(f"Expected one effort limit for {name}, got {count}")
                return match[1] + body + match[3]

            text, count = re.subn(pattern, patch_joint, text, flags=re.DOTALL)
            if count != 1:
                raise ValueError(f"Expected one {name}, got {count}")
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
    arm = patch_dynamics(ARM.read_text(), sides=("right",))
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
