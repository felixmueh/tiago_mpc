#!/usr/bin/env python3
"""Overlay retained reviewed collision geometry on the canonical simulator model.

Only collision elements on links present in the reviewed source are replaced.
Current native joints, inertias, visuals, mimics, loss metadata, comments, and
new sensor links/obstacles remain canonical. No blacklist decisions are made.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / 'models/tiago_pro/tiago_pro.urdf'
SAFETY = ROOT / 'models/safety'
DIRECTORY = SAFETY / 'collision_alpha_wrap_a3_o25_cap001'
REVIEWED = DIRECTORY / 'tiago_pro_collision_alpha_wrap_a3_o25_cap001_reviewed_source.urdf'
ACTIVE = DIRECTORY / 'tiago_pro_collision_alpha_wrap_a3_o25_cap001.urdf'
PROVENANCE = SAFETY / 'provenance.json'


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def parse_xml(payload):
    return ET.fromstring(payload, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))


def generate_active(canonical, reviewed):
    tree, overlay = parse_xml(canonical), parse_xml(reviewed)
    links = {link.get('name'): link for link in tree.findall('link')}
    reviewed_links = overlay.findall('link')
    missing = sorted(link.get('name') for link in reviewed_links if link.get('name') not in links)
    if missing:
        raise ValueError(f'Reviewed collision links absent from canonical model: {missing}')
    replaced = []
    for source in reviewed_links:
        target = links[source.get('name')]
        old_collisions = target.findall('collision')
        insertion = list(target).index(old_collisions[0]) if old_collisions else len(target)
        for collision in old_collisions:
            target.remove(collision)
        for index, collision in enumerate(source.findall('collision')):
            target.insert(insertion + index, deepcopy(collision))
        if old_collisions or source.findall('collision'):
            replaced.append(source.get('name'))
    # Retain the source header, including PAL copyright/license text outside robot.
    header = canonical[:canonical.index(b'<robot')]
    header += (b'<!-- TIAGO_MPC safety: canonical simulator kinematics/inertias with retained '
               b'reviewed alpha-wrap collision elements; regenerate with '
               b'python scripts/prepare_safety_model.py. -->\n')
    payload = header + ET.tostring(tree, encoding='utf-8') + b'\n'
    return payload, {
        'canonical_link_count': len(links),
        'canonical_joint_count': len(tree.findall('joint')),
        'reviewed_link_count': len(reviewed_links),
        'overlaid_collision_links': sorted(replaced),
        'retained_new_collision_links': sorted(
            name for name, link in links.items()
            if name not in {item.get('name') for item in reviewed_links} and link.findall('collision')),
        'active_collision_geometry_count': len(tree.findall('link/collision')),
    }


def outputs():
    canonical, reviewed = CANONICAL.read_bytes(), REVIEWED.read_bytes()
    payload, counts = generate_active(canonical, reviewed)
    provenance = json.loads(PROVENANCE.read_text())
    expected = provenance['sha256'][
        'tiago_configs/collision_alpha_wrap_a3_o25_cap001/tiago_pro_collision_alpha_wrap_a3_o25_cap001.urdf']
    if sha256(reviewed) != expected:
        raise ValueError('Retained reviewed collision source does not match original recorded SHA-256')
    provenance['notes'] = (
        'Historical sha256 keys identify original GATO sources. Reviewed collision source, meshes, '
        'blacklist and alpha-wrap receipt are retained byte for byte. The active collision URDF is '
        'derived from the current canonical simulator model with only reviewed collision elements '
        'overlaid. Historical paths inside receipts are provenance, never runtime fallbacks.')
    provenance['retained_reviewed_urdf'] = str(REVIEWED.relative_to(ROOT))
    provenance['derived_active_model'] = {
        'canonical_model': str(CANONICAL.relative_to(ROOT)),
        'canonical_sha256': sha256(canonical),
        'reviewed_collision_source': str(REVIEWED.relative_to(ROOT)),
        'reviewed_collision_source_sha256': sha256(reviewed),
        'generator': str(Path(__file__).relative_to(ROOT)),
        'generator_sha256': sha256(Path(__file__).read_bytes()),
        'active_model': str(ACTIVE.relative_to(ROOT)),
        'active_sha256': sha256(payload),
        'regenerate_command': 'python scripts/prepare_safety_model.py',
        'blacklist_changes': 'none',
        **counts,
    }
    return {ACTIVE: payload, PROVENANCE: (json.dumps(provenance, indent=2) + '\n').encode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='verify generated model and provenance without writing')
    args = parser.parse_args()
    for path, payload in outputs().items():
        if args.check:
            if path.read_bytes() != payload:
                raise ValueError(f'Run scripts/prepare_safety_model.py to regenerate {path}')
        else:
            path.write_bytes(payload)
    print('Canonical safety model and provenance are current' if args.check else
          'Generated canonical safety model with retained reviewed collision overlay')


if __name__ == '__main__':
    main()
