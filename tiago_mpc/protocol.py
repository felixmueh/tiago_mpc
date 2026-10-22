"""ROS-free messages shared by the I/O process and its MPC worker."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .ros2 import ArmState


@dataclass(frozen=True)
class TaskReference:
    """Position reference in the reduced arm's torso-root frame, metres."""

    target_xyz_m: tuple[float, float, float]
    generation: int = 1
    kind: str = "reach"
    x_span_m: float = 0.28
    y_amplitude_m: float = 0.09
    period_s: float = 24.0
    edge_clearance_m: float = 0.02

    def __post_init__(self):
        if self.kind not in {"reach", "figure8"}:
            raise ValueError("task kind must be reach or figure8")
        xyz = np.asarray(self.target_xyz_m, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("target_xyz_m must contain three finite coordinates")
        if self.generation < 1:
            raise ValueError("task generation must be positive")
        if (not all(math.isfinite(v) for v in (self.x_span_m, self.y_amplitude_m, self.period_s,
                                             self.edge_clearance_m))
                or min(self.x_span_m, self.y_amplitude_m, self.edge_clearance_m) < 0
                or self.period_s <= 0):
            raise ValueError("invalid figure-eight dimensions or period")

    def horizon(self, elapsed_s: float, n: int, dt_s: float) -> np.ndarray:
        if not math.isfinite(elapsed_s) or elapsed_s < 0 or n < 2 or not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("invalid reference time or horizon")
        out = np.zeros((n, 6), dtype=np.float32)
        out[:, :3] = self.target_xyz_m
        if self.kind == "figure8":
            out[:, :3] = figure8_reference(elapsed_s + np.arange(n)*dt_s, self.target_xyz_m,
                dict(period_s=self.period_s, x_span_m=self.x_span_m,
                     y_amplitude_m=self.y_amplitude_m, edge_clearance_m=self.edge_clearance_m))
        return out


def figure8_reference(times, start_xyz, shape):
    """Shared horizontal figure-eight in the reduced arm's root frame."""
    phase = 2*np.pi*np.asarray(times)/shape["period_s"]
    xyz = np.broadcast_to(start_xyz, (len(phase), 3)).copy()
    xyz[:, 0] += shape["edge_clearance_m"] + .5*shape["x_span_m"]*(1-np.cos(phase))
    xyz[:, 1] += shape["y_amplitude_m"]*np.sin(2*phase)
    return xyz


@dataclass(frozen=True)
class SolveRequest:
    request_id: int
    state: ArmState
    task: TaskReference
    task_elapsed_s: float
    deadline_monotonic_sec: float
    dt_elapsed_s: float | None = None


@dataclass(frozen=True)
class WorkerReady:
    ok: bool
    error: str = ""
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SolveReply:
    request_id: int
    generation: int
    source_seq: int
    source_received_monotonic_sec: float
    finished_monotonic_sec: float
    expires_monotonic_sec: float
    # The plan: knot torques u_0..u_{N-2} in native coordinates and their
    # interval. It starts at source_received_monotonic_sec (plan.Plan).
    torques_native: np.ndarray | None = None
    dt_s: float = math.nan
    stats: dict = field(default_factory=dict)
    error: str = ""
