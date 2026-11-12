"""Measure the thesis horizon/batch protocol on the current TIAGo/GATO stack.

Run on Jetson with python -m experiments.jetson_nb_matrix --output _expr_data/jetson-nb.
Use --describe for CPU-only protocol inspection, or --analyze PATH for saved statistics.
Only scalar timings, work and quality statistics are persisted. The simulator and
unshifted lane-zero warm starts stay in memory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import pinocchio as pin
import gato
from gato.common import rk4

from experiments.common import handle_stop_signals
import tiago_mpc as tm


KNOTS = (16, 32, 64, 128, 256, 512)
BATCHES = (1, 2, 4, 8, 16)
HISTORICAL_SOURCE = "04a6ecd1965897cf05dbd3ea3cdb2947da74c8b1"


@dataclass(frozen=True)
class Matrix:
    """The fixed protocol, with optional subsets of its original grid."""
    knots: tuple[int, ...] = KNOTS
    batches: tuple[int, ...] = BATCHES
    repeats: int = 2
    updates: int = 300
    knot_s: float = .03
    control_s: float = .01
    integration_s: float = .001
    timeout_s: float = 180.


def solver_parameters() -> gato.SolverParams:
    """Keep the original scalar tracking objective and its 30 ms/8 ms scaling."""
    scale = .03 / .008
    return gato.SolverParams(max_sqp_iters=1, max_pcg_iters=120, pcg_tol=1e-3,
                             solve_ratio=1., mu=1., rho=.01, adapt_rho=True, linsys="pcg",
                             q_cost=4. * scale, qd_cost=.01 * scale, u_cost=1e-4 * scale,
                             N_cost=80., q_lim_cost=.01 * scale, vel_lim_cost=.001 * scale,
                             ctrl_lim_cost=.003 * scale)


def describe(config: Matrix) -> dict:
    """Describe the study without compiling modules or touching CUDA."""
    return dict(protocol="current-stack-thesis-nb-v1", config=asdict(config),
                historical_protocol_source=HISTORICAL_SOURCE,
                cells=[dict(N=n, batch_size=b, horizon_s=(n - 1) * config.knot_s)
                       for n in config.knots for b in config.batches],
                measured_calls=len(config.knots) * len(config.batches) * config.repeats * config.updates,
                excluded_initialization_calls=len(config.knots) * len(config.batches) * config.repeats,
                solver=asdict(solver_parameters()),
                reference=dict(period_s=24., x_span_m=.32, y_amplitude_m=.08, edge_clearance_m=.02,
                               sample_s=.01, sampling="linear interpolation of periodic 2400-sample reference"),
                model="current canonical TIAGo URDF, native joints, grasp frame, fitted armature; loss-free",
                old_model_changes="historical tool frame, old reduced URDF, q4=-2.35 and zero armature replaced",
                batch_policy="previous unshifted lane-zero plan broadcast to every lane; lane zero commands robot",
                constraints="bare PCG with original soft limit penalties; no ADMM, collision rows, PD or friction feedforward",
                schedule="one excluded solve at t=0; integrate first plan torque for 10 ms; solve at .01 through 3.00 s",
                timing="full-batch native host solve time; wrapper solve-call wall time separately; no solve latency charged",
                recording="statistics only; no states, references, controls, plans, replays or trajectory arrays saved",
                resources="existing native shared-memory opt-in; isolated workers retain unsupported/error/timeout outcomes; no probe kernels",
                qualification="new current-stack cohort, not a numerical reproduction of the historical source/model")


def write_json(path: Path, value) -> None:
    """Atomically retain scalar observations and strict, machine-readable metadata."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(tm.to_dict(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: list[str]) -> str:
    try:
        result = subprocess.run(args, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return "command unavailable"
    return result.stdout + result.stderr


def reference_samples(start: np.ndarray) -> np.ndarray:
    """Construct the original 24-second horizontal reference relative to the current grasp."""
    phase = np.arange(2400) * (2. * np.pi / 2400)
    samples = np.zeros((2400, 6))
    samples[:, :3] = start
    samples[:, 0] += .02 + .16 * (1. - np.cos(phase))
    samples[:, 1] += .08 * np.sin(2. * phase)
    return samples


def reference_at(samples: np.ndarray, timestamp: float, n: int, dt: float = .03) -> np.ndarray:
    """Sample the same periodic interpolation at every horizon length."""
    positions = np.remainder((timestamp + np.arange(n) * dt) / .01, len(samples))
    lower = np.floor(positions).astype(int)
    fraction = (positions - lower)[:, None]
    return (1. - fraction) * samples[lower] + fraction * samples[(lower + 1) % len(samples)]


def hold_seed(x: np.ndarray, n: int, batch: int) -> np.ndarray:
    """Give every lane identical held states and zero torques at initialization."""
    seed = np.zeros((batch, n * 21 - 7), dtype=np.float32)
    for knot in range(n):
        seed[:, knot * 21:knot * 21 + 14] = x
    return seed


def broadcast_seed(xu: np.ndarray, state: np.ndarray, batch: int) -> np.ndarray:
    """Broadcast the previous unshifted lane-zero result and replace only its first state."""
    seed = np.tile(xu[0], (batch, 1))
    seed[:, :14] = state
    return seed


def quality(state: np.ndarray, torque: np.ndarray, reference: np.ndarray) -> dict:
    """Keep scalar executed-state/command quality without retaining the trajectory."""
    bounds = tm.limits()
    q, qd = state[:7], state[7:]
    return dict(tracking_error_m=float(np.linalg.norm(tm.ee_position(q) - reference[0, :3])),
                position_excess_rad=float(max(0., np.max(bounds.lower - q), np.max(q - bounds.upper))),
                speed_excess_rad_s=float(max(0., np.max(np.abs(qd) - bounds.velocity))),
                torque_excess_nm=float(max(0., np.max(np.abs(torque) - bounds.effort))))


def observe(solver, result, wrapper_s: float, update: int, timestamp: float) -> dict:
    """Read actual per-lane work with the full-batch timing boundary."""
    stats = result.stats
    pcg = np.asarray(stats.pcg_iters, dtype=int)
    sqp = np.asarray(stats.sqp_iters, dtype=int)
    native_ms = float(result.solve_time_us) / 1000.
    if not np.isfinite([native_ms, wrapper_s]).all() or min(native_ms, wrapper_s) <= 0:
        raise ValueError("nonfinite or nonpositive solve timing")
    if sqp.shape != (solver.batch_size,) or np.any(sqp < 0) or np.any(sqp > 1):
        raise ValueError("native SQP telemetry differs from the one-iteration cap")
    if pcg.ndim != 2 or pcg.shape[1] != solver.batch_size or np.any(pcg < 0):
        raise ValueError("PCG telemetry does not cover every lane")
    return dict(update=update, simulation_s=timestamp, native_ms=native_ms, wrapper_ms=wrapper_s * 1000,
                sqp_mean=float(sqp.mean()), sqp_min=int(sqp.min()), sqp_max=int(sqp.max()),
                sqp_lane_sum=int(sqp.sum()), pcg_sum=int(pcg.sum()), pcg_slots=int(pcg.size),
                pcg_cap_count=int(np.sum(pcg >= solver.params.max_pcg_iters)),
                finite_plan=bool(np.isfinite(result.xu).all()),
                initial_merit_mean=float(np.mean(stats.initial_merit)) if np.isfinite(stats.initial_merit).all() else None,
                final_merit_mean=float(np.mean(stats.final_merit)) if np.isfinite(stats.final_merit).all() else None)


def run_updates(solver, advance, stream, config: Matrix) -> dict:
    """Run one initialization and the fixed count, streaming no trajectory values."""
    state = np.r_[tm.Q_START, np.zeros(7)]
    samples = reference_samples(tm.ee_position(state[:7]))
    seed = hold_seed(state, solver.N, solver.batch_size)
    reference = reference_at(samples, 0., solver.N, config.knot_s)
    solver.reset_dual()
    initialization = None
    for update in range(config.updates + 1):
        if tm.STOP.is_set():
            return dict(status="interrupted", samples=max(0, update - 1), initialization=initialization)
        timestamp = update * config.control_s
        if update:
            state = advance(state, result.xu[0, 14:21], config.control_s, config.integration_s)
            if not np.isfinite(state).all():
                return dict(status="nonfinite_state", samples=update - 1, initialization=initialization,
                            failure_update=update)
            seed = broadcast_seed(result.xu, state, solver.batch_size)
            reference = reference_at(samples, timestamp, solver.N, config.knot_s)
        states = np.tile(state, (solver.batch_size, 1))
        references = np.tile(reference.reshape(-1), (solver.batch_size, 1))
        solver.reset_rho()
        started = time.perf_counter()
        result = solver.solve(states, references, seed, budget_s=None)
        wrapper_s = time.perf_counter() - started
        row = observe(solver, result, wrapper_s, update, timestamp)
        row["initialization"] = update == 0
        if row["finite_plan"]:
            row.update(quality(state, result.xu[0, 14:21], reference))
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        if not update:
            initialization = row
        if not row["finite_plan"]:
            return dict(status="nonfinite_plan", samples=update, initialization=initialization,
                        failure_update=update)
    return dict(status="completed", samples=config.updates, initialization=initialization)


def load_solver(n: int, batch: int, binary: Path):
    """Load the exact prepared module and construct bare PCG without controller-added ADMM."""
    module_name = binary.name.split(".")[0]
    prefix = f"bsqpN{n}_"
    if not module_name.startswith(prefix):
        raise ValueError("binary identity does not match the requested horizon")
    name = "gato." + module_name
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, binary)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load prepared module {binary}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    elif Path(sys.modules[name].__file__).resolve() != binary.resolve():
        raise ValueError("a different binary with the same module name is already loaded")
    plant = module_name[len(prefix):]
    solver = gato.BSQP(str(gato.robot_urdf(plant)), batch, n, .03, params=solver_parameters(), plant_type=plant)
    solver.set_collect_stats(True)
    if getattr(solver, "admm_active", False):
        raise ValueError("the matrix requires bare PCG without ADMM rows")
    return solver


def run_worker(out: Path, n: int, batch: int, repeat: int, binary: Path) -> int:
    """Isolate CUDA failures to one repeat and keep already streamed statistics."""
    folder = out / f"N{n}_B{batch}_R{repeat}"
    folder.mkdir(parents=True, exist_ok=False)
    receipt = dict(N=n, batch_size=batch, repeat=repeat, status="starting", samples=0,
                   binary_sha256=file_sha(binary), solver=asdict(solver_parameters()))
    write_json(folder / "summary.json", receipt)
    try:
        solver = load_solver(n, batch, binary)
        model = pin.buildModelFromUrdf(str(tm.arm_urdf()))
        model.armature = np.asarray(tm.ARMATURE_KG_M2)
        model.gravity.linear = np.array([0., 0., -9.81])
        data = model.createData()
        def advance(state: np.ndarray, torque: np.ndarray, duration: float, step: float):
            q, qd = state[:7].copy(), state[7:].copy()
            elapsed = 0.
            while elapsed < duration - 1e-14:
                dt = min(step, duration - elapsed)
                q, qd = rk4(model, data, q, qd, torque, dt)
                elapsed += dt
                if not np.isfinite(q).all() or not np.isfinite(qd).all():
                    break
            return np.r_[q, qd]
        receipt.update(status="running", resolved_solver=asdict(solver.params),
                       model_sha256=file_sha(tm.arm_urdf()), frame=tm.GRASP_FRAME,
                       armature=list(tm.ARMATURE_KG_M2))
        write_json(folder / "summary.json", receipt)
        with (folder / "solves.jsonl").open("w", buffering=1) as stream:
            receipt.update(run_updates(solver, advance, stream, Matrix(knots=(n,), batches=(batch,))))
    except KeyboardInterrupt:
        receipt.update(status="interrupted")
    except Exception:
        error = traceback.format_exc()
        failure = worker_failure(1, error)
        receipt.update(status="error" if failure == "worker_error" else failure, error=error)
    write_json(folder / "summary.json", receipt)
    return 0 if receipt["status"] == "completed" else 1


def summarize(values: np.ndarray) -> dict:
    return dict(mean_ms=float(values.mean()), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), p99_ms=float(np.percentile(values, 99)),
                sd_ms=float(values.std()), variance_ms2=float(values.var()),
                min_ms=float(values.min()), max_ms=float(values.max()))


def read_rows(path: Path) -> list[dict]:
    """Read even a worker killed after its last complete streamed observation."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return rows


def analyze(out: Path) -> None:
    """Export every planned cell, leaving failed-cell headline timing fields empty."""
    study = json.loads((out / "study.json").read_text())
    config = Matrix(**{key: tuple(value) if key in ("knots", "batches") else value
                       for key, value in study["config"].items()})
    attempts = {(case["N"], case["batch_size"], case["repeat"]): case for case in study["attempts"]}
    cells, calls, repeats = [], [], []
    for n in config.knots:
        for batch in config.batches:
            samples, statuses = [], []
            for repeat in range(1, config.repeats + 1):
                case = attempts.get((n, batch, repeat), dict(status="not_run"))
                rows = read_rows(out / f"N{n}_B{batch}_R{repeat}" / "solves.jsonl")
                measured = [row for row in rows if not row["initialization"]]
                for row in rows:
                    calls.append(dict(N=n, batch_size=batch, repeat=repeat, **row))
                finite = all(row["finite_plan"] for row in measured) if measured else None
                complete = case["status"] == "completed" and len(measured) == config.updates and finite
                status = "completed" if complete else case["status"] if case["status"] != "completed" else "incomplete"
                statuses.append(status)
                repeats.append(dict(N=n, batch_size=batch, repeat=repeat, status=status,
                                    measured_samples=len(measured), finite_plans=finite,
                                    initialization_samples=sum(row["initialization"] for row in rows)))
                samples.extend(measured)
            complete = all(status == "completed" for status in statuses)
            status = "completed" if complete else ";".join(sorted(set(statuses)))
            cell = dict(N=n, batch_size=batch, horizon_s=(n - 1) * config.knot_s, status=status,
                        measured_samples=len(samples), completed_repeats=statuses.count("completed"),
                        expected_samples=config.updates * config.repeats)
            for boundary in ("native_ms", "wrapper_ms"):
                for key in ("mean_ms", "median_ms", "p95_ms", "p99_ms", "sd_ms", "variance_ms2", "min_ms", "max_ms"):
                    cell[boundary.removesuffix("_ms") + "_" + key] = None
                if complete:
                    cell.update({boundary.removesuffix("_ms") + "_" + key: value
                                 for key, value in summarize(np.array([row[boundary] for row in samples])).items()})
            if samples:
                slots = sum(row["pcg_slots"] for row in samples)
                cell.update(pcg_cap_share=sum(row["pcg_cap_count"] for row in samples) / slots if slots else None,
                            mean_sqp=float(np.mean([row["sqp_mean"] for row in samples])),
                            tracking_rms_m=float(np.sqrt(np.mean([row.get("tracking_error_m", np.nan) ** 2
                                                               for row in samples]))),
                            max_position_excess_rad=max(row.get("position_excess_rad", 0.) for row in samples),
                            max_speed_excess_rad_s=max(row.get("speed_excess_rad_s", 0.) for row in samples),
                            max_command_excess_nm=max(row.get("torque_excess_nm", 0.) for row in samples))
                if not np.isfinite(cell["tracking_rms_m"]):
                    cell["tracking_rms_m"] = None
            cells.append(cell)
    for filename, rows in (("cells.csv", cells), ("repeat-status.csv", repeats), ("solve-times.csv", calls)):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (out / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    write_json(out / "cells.json", cells)
    counts = dict(Counter(cell["status"] for cell in cells))
    write_json(out / "completion.json", dict(planned_cells=len(cells), outcomes=counts,
               completed_cells=counts.get("completed", 0), all_completed=all(c["status"] == "completed" for c in cells)))
    report = ["# Current-stack Jetson N × B matrix", "", json.dumps(counts), "",
              "This is a new current-stack cohort using the thesis matrix protocol. It uses the current canonical "
              "TIAGo URDF, native joints, grasp frame and fitted armature instead of the historical model/tool frame. "
              "Source/binary identities are saved; do not interpret differences from old results as a GPU-only effect.", "",
              "N16…512 × B1/2/4/8/16, SQP1, PCG120, 30 ms knots, 10 ms simulated updates and RK4 substeps up to 1 ms. "
              "Two fresh repeats have one excluded initialization at t=0 and 300 measured solves at 0.01…3.00 s. "
              "Every lane receives the previous unshifted lane-zero plan. Lane-zero first-knot torque commands the "
              "loss-free simulator. No ADMM, PD, friction feedforward, wall budget or slow-solve cutoff.", "",
              "The original 24-second horizontal figure-eight and 30 ms/8 ms running/soft-limit cost scaling are retained; "
              "terminal position weight remains 80. Only scalar timing, work and quality statistics are saved.", "",
              "Primary native host timings cover a full batch and exclude initialization, Python work and integration. "
              "Wrapper solve-call times are exported separately. They are not end-to-end control rates or per-candidate "
              "latencies. Cell timing summaries require both complete repeats. Partial/failure rows remain in "
              "solve-times.csv and worker solves.jsonl; all planned cells remain in cells.csv.", "",
              "Quality is sampled at 10 ms update times; torque excess describes the next unmodified command. "
              "No command clipping, residual/KKT acceptance gate or trajectory safety certificate is implied. "
              "SD/variance describe serial observed spread, not uncertainty across independent repeats.", "",
              "No retries occur. Capacity/build errors and worker timeouts are retained with logs. The current "
              "PCG kernel requests dynamic shared-memory opt-in above 48 KiB; actual Jetson support is determined "
              "during execution. No resource probe launches or local GTX measurements were used to prepare this study.", ""]
    (out / "REPORT.md").write_text("\n".join(report))
    plot_heatmaps(out, config, cells)


def plot_heatmaps(out: Path, config: Matrix, cells: list[dict]) -> None:
    """Show supported full-batch mean and median native times with failed cells labelled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for field in ("native_mean_ms", "native_median_ms"):
        values = np.full((len(config.knots), len(config.batches)), np.nan)
        for i, n in enumerate(config.knots):
            for j, batch in enumerate(config.batches):
                row = next(cell for cell in cells if cell["N"] == n and cell["batch_size"] == batch)
                if row[field] is not None:
                    values[i, j] = row[field]
        fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
        mesh = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap="viridis")
        for i, n in enumerate(config.knots):
            for j, batch in enumerate(config.batches):
                row = next(cell for cell in cells if cell["N"] == n and cell["batch_size"] == batch)
                label = f"{values[i, j]:.2f}" if np.isfinite(values[i, j]) else row["status"]
                ax.text(j, i, label, ha="center", va="center", fontsize=7,
                        color="white" if np.isfinite(values[i, j]) else "black")
        ax.set(xticks=range(len(config.batches)), xticklabels=config.batches,
               yticks=range(len(config.knots)), yticklabels=config.knots,
               xlabel="Batch size", ylabel="Horizon knots", title=field.replace("_", " ") + " · current stack · SQP1")
        fig.colorbar(mesh, ax=ax, label="Full-batch native solve time (ms)")
        fig.savefig(out / f"{field}.png", dpi=160)
        fig.savefig(out / f"{field}.pdf")
        plt.close(fig)


def worker_failure(returncode: int, log: str) -> str:
    """Only name capacity failures when the retained diagnostic identifies that cause."""
    if "out of memory" in log.lower():
        return "unsupported_memory"
    if "too many resources requested" in log.lower() or "uses too much shared data" in log.lower():
        return "unsupported_resources"
    if "GPUassert:" in log and "cuda.cuh" in log and "invalid argument" in log.lower():
        return "unsupported_shared_memory_optin"
    return "worker_error" if returncode else "incomplete"


def run_matrix(out: Path, config: Matrix) -> None:
    """Build once per horizon and run sequential isolated repeats without automatic retries."""
    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") != "0":
        raise ValueError("unset CUDA_LAUNCH_BLOCKING before benchmarking")
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    source.mkdir()
    source.joinpath(Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    root = Path(__file__).resolve().parents[1]
    source.joinpath("tracked.diff").write_text(command(["git", "-C", str(root), "diff", "HEAD"]))
    study = describe(config)
    study.update(provenance=tm.provenance(), machine=platform.uname()._asdict(),
                 gpu_before=command(["nvidia-smi"]), attempts=[], binaries={},
                 jetson_mode="MODE_40W (user-confirmed; not queried)")
    write_json(out / "study.json", study)
    for n in config.knots:
        if tm.STOP.is_set():
            break
        try:
            params = replace(tm.TIAGO, N=n, dt_s=config.knot_s, cost=tm.default_cost(n),
                             collision=None, friction=None, max_sqp_time_s=None)
            binary = tm.compile_binary(params, jobs=1)
            study["binaries"][str(n)] = dict(path=str(binary), sha256=file_sha(binary))
        except Exception:
            error = traceback.format_exc()
            (out / f"N{n}-build-error.log").write_text(error)
            for batch in config.batches:
                for repeat in range(1, config.repeats + 1):
                    study["attempts"].append(dict(N=n, batch_size=batch, repeat=repeat, status="build_error", error=error))
            write_json(out / "study.json", study)
            continue
        write_json(out / "study.json", study)
        for batch in config.batches:
            for repeat in range(1, config.repeats + 1):
                if tm.STOP.is_set():
                    break
                label = f"N{n}_B{batch}_R{repeat}"
                log_path = out / f"{label}.log"
                args = [sys.executable, "-m", "experiments.jetson_nb_matrix", "--output", str(out),
                        "--cell", str(n), str(batch), str(repeat), "--binary", str(binary)]
                env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                print(f"Starting {label}", flush=True)
                status = None
                started = time.time()
                with log_path.open("w") as log:
                    try:
                        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT,
                                                timeout=config.timeout_s, env=env, check=False, cwd=root)
                    except subprocess.TimeoutExpired:
                        status = "timeout"
                receipt_path = out / label / "summary.json"
                receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
                if status is None:
                    status = receipt.get("status", worker_failure(result.returncode, log_path.read_text()))
                    if status in ("starting", "running"):
                        status = worker_failure(result.returncode, log_path.read_text())
                study["attempts"].append(dict(N=n, batch_size=batch, repeat=repeat, status=status,
                                              started_unix_s=started, ended_unix_s=time.time()))
                write_json(out / "study.json", study)
                print(f"Finished {label}: {status}", flush=True)
                if status == "interrupted":
                    tm.STOP.set()
    study["gpu_after"] = command(["nvidia-smi"])
    write_json(out / "study.json", study)
    analyze(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--analyze", type=Path)
    parser.add_argument("--describe", "--dry-run", action="store_true")
    parser.add_argument("--knots", nargs="+", type=int, choices=KNOTS, default=KNOTS)
    parser.add_argument("--batches", nargs="+", type=int, choices=BATCHES, default=BATCHES)
    parser.add_argument("--cell", nargs=3, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--binary", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = Matrix(knots=tuple(args.knots), batches=tuple(args.batches))
    if len(set(config.knots)) != len(config.knots) or len(set(config.batches)) != len(config.batches):
        parser.error("knots and batches must not contain duplicates")
    if sum((args.output is not None, args.analyze is not None, args.describe)) != 1:
        parser.error("choose exactly one of --output, --analyze or --describe")
    if args.cell and (args.binary is None or args.output is None):
        parser.error("a worker requires --binary and --output")
    handle_stop_signals()
    if args.describe:
        print(json.dumps(describe(config), indent=2))
    elif args.analyze:
        analyze(args.analyze)
    elif args.cell:
        return run_worker(args.output.resolve(), *args.cell, args.binary.resolve())
    else:
        run_matrix(args.output.resolve(), config)
        if tm.STOP.is_set():
            return 130
        completion = json.loads((args.output / "completion.json").read_text())
        return 0 if completion["all_completed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
