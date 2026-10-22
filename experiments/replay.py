"""The visualizer's replay export; NumPy and the standard library only.

save_replay writes the visualizer's portable NPZ v1 directly from native q
samples. It does not import or run GATO, Pinocchio, or the visualizer (the
TIAGo names come from tiago_mpc.tiago, which loads only NumPy).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from tiago_mpc.tiago import FLANGE_FRAME, FULL_URDF, GRASP_FRAME, RIGHT_ARM_JOINTS, ROOT_FRAME

FORMAT = "tiago-replay"
VERSION = 1


def export_replay(record):
    """Write <record.out>/replay.npz on execution time: the state and command at each
    tick, the task's reference and every plan; nothing for fewer than two ticks."""
    trajectory = record.trajectory
    time_s, state, command = trajectory["time_s"], trajectory["state"], trajectory["command"]
    if len(time_s) < 2:
        return None
    sent = np.isfinite(command).all(axis=1)
    dt, n = record.params.dt_s, record.params.N
    plans = [dict(origin_time_s=float(t0), time_s=np.arange(n) * dt, q=states[:, :7],
                  qd_rad_s=states[:, 7:], torque_nm=torques, torque_time_s=np.arange(n - 1) * dt)
             for t0, states, torques in zip(trajectory["plan_t0_s"], trajectory["plan_states"],
                                            trajectory["plan_torques"])]
    return save_replay(record.out / "replay.npz", time_s=time_s, q=state[:, :7], qd_rad_s=state[:, 7:],
                       joint_names=RIGHT_ARM_JOINTS, time_basis=record.plant.settings["clock"],
                       torque_nm=command[sent] if sent.any() else None,
                       torque_time_s=time_s[sent] if sent.any() else None,
                       reference_xyz_m=np.array([record.task.at(t, 2, dt)[0][0, :3] for t in time_s]),
                       reference_time_s=time_s, reference_frame=ROOT_FRAME, plans=plans,
                       end_effector_frame=record.params.ee_frame, label=record.out.name)


def model_sha256(model_urdf):
    return hashlib.sha256(Path(model_urdf).read_bytes()).hexdigest()


def _finite_json(value, name):
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must contain finite JSON values: {exc}") from exc


def _series(time_s, q, name):
    times, positions = np.asarray(time_s), np.asarray(q)
    if (times.dtype.kind not in "fiu" or times.ndim != 1 or len(times) < 2 or
            not np.isfinite(times).all() or times[0] < 0 or not (times[1:] > times[:-1]).all()):
        raise ValueError(f"{name} timestamps must be finite, nonnegative, strictly increasing, with at least two samples")
    if (positions.dtype.kind not in "fiu" or positions.shape != (len(times), 7) or
            not np.isfinite(positions).all()):
        raise ValueError(f"{name} q must be a finite real (T, 7) native-coordinate array")


def _reference(xyz, times, frame):
    if xyz is None:
        if times is not None:
            raise ValueError("Reference times require reference positions")
        return
    if frame not in ("base_footprint", ROOT_FRAME):
        raise ValueError(f"Reference positions require reference_frame='base_footprint' or '{ROOT_FRAME}'")
    points = np.asarray(xyz)
    if points.dtype.kind not in "fiu" or not np.isfinite(points).all():
        raise ValueError("Reference positions must be finite real numbers")
    if times is None:
        if points.shape != (3,):
            raise ValueError("A static reference must have shape (3,)")
    else:
        stamps = np.asarray(times)
        if (stamps.dtype.kind not in "fiu" or stamps.ndim != 1 or len(stamps) < 1 or
                not np.isfinite(stamps).all() or stamps[0] < 0 or not (stamps[1:] > stamps[:-1]).all() or
                points.shape != (len(stamps), 3)):
            raise ValueError("Moving reference requires increasing finite times (R,) and points (R, 3)")

def _telemetry(time_s, qd_rad_s, torque_nm, torque_time_s, name):
    times = np.asarray(time_s)
    if qd_rad_s is not None:
        velocity = np.asarray(qd_rad_s)
        if velocity.dtype.kind not in "fiu" or velocity.shape != (len(times), 7) or not np.isfinite(velocity).all():
            raise ValueError(f"{name} qd_rad_s must be a finite real (T, 7) array")
    if torque_nm is None:
        if torque_time_s is not None:
            raise ValueError(f"{name} torque timestamps require torque_nm")
        return
    torque = np.asarray(torque_nm)
    stamps = times if torque_time_s is None else np.asarray(torque_time_s)
    if (stamps.dtype.kind not in "fiu" or stamps.ndim != 1 or len(stamps) < 1 or
            not np.isfinite(stamps).all() or not (stamps[1:] > stamps[:-1]).all() or
            stamps[0] < times[0] or stamps[-1] > times[-1]):
        raise ValueError(f"{name} torque_time_s must increase within the trajectory time range")
    if torque.dtype.kind not in "fiu" or torque.shape != (len(stamps), 7) or not np.isfinite(torque).all():
        raise ValueError(f"{name} torque_nm must be finite real (K, 7), with torque_time_s for K != T")


def _joint_limits(limits):
    if limits is None:
        return
    fields = ("lower_position_rad", "upper_position_rad", "velocity_rad_s", "effort_nm")
    if not isinstance(limits, dict) or any(field not in limits for field in fields):
        raise ValueError("joint_limits requires lower_position_rad, upper_position_rad, velocity_rad_s and effort_nm")
    for field in fields:
        values = np.asarray(limits[field])
        if values.dtype.kind not in "fiu" or values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError(f"joint_limits.{field} must contain seven finite real values")
    if (not (np.asarray(limits["upper_position_rad"]) > limits["lower_position_rad"]).all() or
            not (np.asarray(limits["velocity_rad_s"]) > 0).all() or not (np.asarray(limits["effort_nm"]) > 0).all()):
        raise ValueError("joint_limits needs ordered position bounds and positive velocity/effort limits")
    if "source" in limits and (not isinstance(limits["source"], str) or not limits["source"].strip()):
        raise ValueError("joint_limits.source must be a nonempty string")


def save_replay(path, *, time_s, q, joint_names, model_urdf=FULL_URDF, time_basis,
                label=None, fixed_joints=None, reference_xyz_m=None,
                reference_time_s=None, reference_frame=None, plans=None, provenance=None,
                qd_rad_s=None, torque_nm=None, torque_time_s=None, joint_limits=None,
                end_effector_frame=FLANGE_FRAME):
    """Export one compressed v1 file using a temporary file and atomic rename.

    Each optional plan is a dict containing origin_time_s, time_s and q;
    optional keys: id, label, stats, reference_xyz_m, reference_time_s,
    reference_frame. Plan times/reference times are offsets from its origin.
    Every q array uses the supplied native joint_names order. Existing output
    paths are replaced atomically; never pass an original recording as output.
    Optional provenance is a finite JSON object preserving source/model details.
    qd_rad_s aligns state samples. torque_nm is native applied actuator torque,
    held from torque_time_s until the next timestamp (defaults to state times).
    For interval controls pass torque_time_s=time_s[:-1]; at the final state the
    viewer shows the last applied interval. Plans use predicted torque and omit
    the terminal control. Optional joint_limits use the same joint_names order.
    end_effector_frame identifies the point tracked by the reference. The legacy
    flange default preserves old exports; new grasp-center runs must pass their
    recorded frame explicitly (e.g. summary['end_effector_frame']).
    """
    if (not isinstance(time_basis, str) or time_basis not in ("simulation", "wall")):
        raise ValueError("time_basis must be simulation or wall")
    names = list(joint_names)
    if (len(names) != 7 or any(not isinstance(n, str) for n in names) or
            len(set(names)) != 7 or set(names) != set(RIGHT_ARM_JOINTS)):
        raise ValueError("joint_names must contain each of the seven native right-arm joints once")
    _series(time_s, q, "Recorded")
    _reference(reference_xyz_m, reference_time_s, reference_frame)
    _telemetry(time_s, qd_rad_s, torque_nm, torque_time_s, "Recorded")
    _joint_limits(joint_limits)
    if end_effector_frame not in (FLANGE_FRAME, GRASP_FRAME):
        raise ValueError("Unsupported end_effector_frame")
    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".npz":
        raise ValueError("Replay export path must end in .npz")
    metadata = dict(format=FORMAT, version=VERSION, joint_names=names,
                    model_sha256=model_sha256(model_urdf), time_basis=time_basis,
                    end_effector_frame=end_effector_frame)
    arrays = dict(time_s=np.asarray(time_s), q=np.asarray(q))
    for key, value in (("qd_rad_s", qd_rad_s), ("torque_nm", torque_nm), ("torque_time_s", torque_time_s)):
        if value is not None:
            arrays[key] = np.asarray(value)
    if joint_limits is not None:
        metadata["joint_limits"] = {key: (np.asarray(value).tolist() if key != "source" else value)
                                    for key, value in joint_limits.items()}

    if provenance is not None:
        if not isinstance(provenance, dict):
            raise ValueError("provenance must be a JSON object")
        metadata["provenance"] = provenance
    if label is not None:
        metadata["label"] = label
    if fixed_joints is not None:
        metadata["fixed_joints"] = fixed_joints
    if reference_frame is not None:
        metadata["reference_frame"] = reference_frame
    if reference_xyz_m is not None:
        arrays["reference_xyz_m"] = np.asarray(reference_xyz_m)
    if reference_time_s is not None:
        arrays["reference_time_s"] = np.asarray(reference_time_s)
    if plans is not None:
        metadata["predictions"] = []
        for index, plan in enumerate(plans):
            origin = plan.get("origin_time_s")
            if isinstance(origin, bool) or not isinstance(origin, (int, float)) or not np.isfinite(origin) or origin < 0:
                raise ValueError("Each prediction requires a finite nonnegative origin_time_s")
            _series(plan["time_s"], plan["q"], f"Prediction {index}")
            _reference(plan.get("reference_xyz_m"), plan.get("reference_time_s"),
                       plan.get("reference_frame", reference_frame))
            _telemetry(plan["time_s"], plan.get("qd_rad_s"), plan.get("torque_nm"),
                       plan.get("torque_time_s"), f"Prediction {index}")
            prefix = f"plan_{index:04d}"
            descriptor = dict(origin_time_s=plan.get("origin_time_s"),
                              time_key=prefix+"_time_s", q_key=prefix+"_q")
            arrays[descriptor["time_key"]] = np.asarray(plan["time_s"])
            arrays[descriptor["q_key"]] = np.asarray(plan["q"])
            for key in ("id", "label", "stats", "reference_frame"):
                if key in plan:
                    descriptor[key] = plan[key]
            for key, field in (("qd_rad_s", "qd_key"), ("torque_nm", "torque_key"),
                               ("torque_time_s", "torque_time_key"),
                               ("reference_xyz_m", "reference_xyz_key"),
                               ("reference_time_s", "reference_time_key")):
                if plan.get(key) is not None:
                    descriptor[field] = prefix+"_"+key
                    arrays[descriptor[field]] = np.asarray(plan[key])
            metadata["predictions"].append(descriptor)
    _finite_json(metadata, "Replay metadata")
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, allow_nan=False))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="."+destination.name+".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
