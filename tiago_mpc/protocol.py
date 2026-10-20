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
    x_span_m: float = 0.24
    y_amplitude_m: float = 0.08
    period_s: float = 24.0

    def __post_init__(self):
        if self.kind not in {"reach", "figure8"}:
            raise ValueError("task kind must be reach or figure8")
        xyz = np.asarray(self.target_xyz_m, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("target_xyz_m must contain three finite coordinates")
        if self.generation < 1:
            raise ValueError("task generation must be positive")
        if (not all(math.isfinite(v) for v in (self.x_span_m, self.y_amplitude_m, self.period_s))
                or self.x_span_m < 0 or self.y_amplitude_m < 0 or self.period_s <= 0):
            raise ValueError("invalid figure-eight dimensions or period")

    def horizon(self, elapsed_s: float, n: int, dt_s: float) -> np.ndarray:
        if not math.isfinite(elapsed_s) or elapsed_s < 0 or n < 2 or not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("invalid reference time or horizon")
        out = np.zeros((n, 6), dtype=np.float32)
        out[:, :3] = self.target_xyz_m
        if self.kind == "figure8":
            phase = 2 * np.pi * (elapsed_s + np.arange(n) * dt_s) / self.period_s
            out[:, 0] += 0.5 * self.x_span_m * (1 - np.cos(phase))
            out[:, 1] += self.y_amplitude_m * np.sin(2 * phase)
        return out


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
    u_native: np.ndarray | None = None
    stats: dict = field(default_factory=dict)
    error: str = ""
