#!/usr/bin/env python3
"""Apply supplied arm parameters and add the reduced arm's nominal grasp frame.

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
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SOURCE = MODELS / "tiago_pro/tiago_pro_sim_source.urdf"
FULL = MODELS / "tiago_pro/tiago_pro.urdf"
ARM = MODELS / "tiago_right/tiago_right_arm.urdf"
# Midpoint of the two fingertip contact-pad centers at zero gripper posture,
# derived from the simulator URDF and fingertip.stl. This is a nominal geometric
# TCP, not a hardware calibration. Keep the flange and all inertias unchanged.
GRASP_FRAME = '''  <!-- Nominal pad-center TCP at zero gripper posture; not hardware-calibrated. -->
  <link name="arm_right_grasp_link" />
  <joint name="arm_right_grasp_joint" type="fixed">
    <parent link="arm_right_tool_link" />
    <child link="arm_right_grasp_link" />
    <origin rpy="0 0 0" xyz="0 0 0.139288457525" />
  </joint>
'''
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
# Physical right-arm fit of 2026-10-16 (figure-eight runs; residual torque
# against the loss-free RNEA, b*v + c*sign(v) + o for |v| > .02 rad/s; pooled
# "all_fig8_physical" in thesis research/physical-tiago-20261016/model-mismatch/
# data/fits.json). Columns: Coulomb Nm, viscous Nm s/rad. Joints 5 to 7 barely
# moved, so they have no fit.
IDENTIFIED_RIGHT = {1: (2.01, 4.44), 2: (2.39, 5.07), 3: (2.03, 2.87), 4: (1.91, 2.82)}
# Written into both URDFs before <robot>: outside the root element, so the
# generated GPU copy and GATO's codegen key do not change. XML comments must
# not contain "--".
CAUTION = '''<!-- TIAGO_MPC caution (2026-10-22): the arm friction, damping and armature
     values below are QUESTIONABLE. They are the supplied table of 2026-10-21,
     not an identification, and no public PAL file contains the armature.
     Right arm, physical fit of 2026-10-16 (joints 1 to 4): Coulomb 2.01, 2.39,
     2.03, 1.91 Nm (1.2 to 2.1x the supplied values), viscous 4.44, 5.07, 2.87,
     2.82 Nm s/rad (0.5 to 0.94x), breakaway 10 to 25 % above Coulomb. Joints 5
     to 7 and the left arm are not identified.
     Stiffness: with link-only inertia (joint 7: 0.00077 kg m^2) the supplied
     damping gives joint 7 a 0.26 ms time constant. Explicit integration then
     needs steps below 0.45 ms (RK4, worst pose) or knots below 0.32 ms
     (GATO's explicit update); with the armature active the limits are 114 ms
     and 82 ms. See tiago_mpc_visualizer/docs/friction.md and, in the thesis
     workspace, research/armature-support-and-stiff-joint-damping.md. -->
'''


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
                    if side == "right" and i in IDENTIFIED_RIGHT:
                        fit = ('physical fit 2026-10-16: friction {0} Nm, damping {1} Nm s/rad'
                               .format(*IDENTIFIED_RIGHT[i]))
                    else:
                        fit = 'not identified on the physical arm'
                    # One comment per joint, as before: the GPU model is written
                    # from this file, and GATO's codegen key hashes its bytes,
                    # including the whitespace a removed comment leaves behind.
                    comment = (f'{indent}<!-- TIAGO_MPC dynamics: QUESTIONABLE values supplied by '
                               f'Felix 2026-10-21 ({fit}; see the caution above <robot>); '
                               'friction Nm, damping Nm s/rad; armature kg m^2 '
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


def add_caution(text):
    """Put the parameter caution right before <robot>, replacing an earlier one."""
    text = re.sub(r'(?s)<!-- TIAGO_MPC caution.*?-->\r?\n', '', text)
    text, count = re.subn(r'(?m)^(?=<robot\b)', lambda m: CAUTION, text, count=1)
    if count != 1:
        raise ValueError("Expected one <robot> element")
    return text


def asset_manifest(full, arm):
    paths = [SOURCE, FULL, ARM, *sorted((MODELS / "tiago_pro/meshes").glob("*.stl"))]
    replacements = {FULL: full.encode(), ARM: arm.encode()}
    return {str(p.relative_to(MODELS)): dict(
        sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
        for p in paths for payload in [replacements.get(p, p.read_bytes())]}


def add_grasp_frame(text):
    """Add a massless application frame without rewriting the retained model."""
    root = ET.fromstring(text)
    expected = ET.fromstring('<robot>' + GRASP_FRAME + '</robot>')
    for element in expected:
        found = root.find(f'{element.tag}[@name="{element.get("name")}"]')
        if found is not None and ET.tostring(found).strip() != ET.tostring(element).strip():
            raise ValueError(f"Unexpected existing grasp frame: {element.get('name')}")
    present = [root.find(f'{e.tag}[@name="{e.get("name")}"]') is not None for e in expected]
    if all(present):
        return text
    if any(present):
        raise ValueError("Incomplete existing grasp frame")
    return text.replace('</robot>', GRASP_FRAME + '</robot>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify prepared models/manifest without writing")
    args = parser.parse_args()
    full = add_caution(patch_dynamics(SOURCE.read_text()))
    arm = add_caution(add_grasp_frame(patch_dynamics(ARM.read_text(), sides=("right",))))
    manifest = json.dumps(asset_manifest(full, arm), indent=2, sort_keys=True) + '\n'
    outputs = {FULL: full, ARM: arm, MODELS / "manifest.json": manifest}
    for path, expected in outputs.items():
        if args.check:
            if path.read_bytes() != expected.encode():
                raise ValueError(f"Run scripts/prepare_models.py to update {path}")
        else:
            path.write_text(expected)
    print("Prepared model metadata and manifest are current" if args.check else
          "Prepared full model, reduced arm metadata/grasp frame and manifest")


if __name__ == "__main__":
    main()
