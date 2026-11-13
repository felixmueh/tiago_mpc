"""Compare one-SQP PCG/BDSV timings with plant friction, direct torque and unshifted warm starts.

Run: python -m experiments.pcg_bdsv_histograms --output _expr_data/pcg-bdsv-jetson.
Analyze without GPU work: python -m experiments.pcg_bdsv_histograms --analyze PATH.
No trajectories, command ticks, plans or replays are written.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from experiments.common import handle_stop_signals
from experiments.figure8 import FIGURE8, Figure8
from experiments.reach import Reach
import tiago_mpc as tm


SOLVERS = ("pcg", "bdsv")
TASKS = ("figure8",)
KNOTS = (16, 32, 64, 128, 256, 512)
SETTINGS = tm.VirtualTimeSettings.ideal(command_period=.002, update_period=.05)
DURATION_S = 4.
BARRIER = dict(mu=3e-3, delta=.05)
PROTOCOL = "jetson-b1-barrier-pcg-bdsv-sqp1-figure8-4s-20hz-stages-v6"
STAGE_NAMES = ("kkt_us", "schur_us", "linsys_stage_us", "merit_us", "line_search_us")
DIAGNOSTICS = (*STAGE_NAMES, "initial_merit_us", "linear_solve_us", "step_size", "line_search_ran",
               "completed_unix_s", "batch_size")
SCALAR_STATS = ("solve_time_us", "wall_s", "iterations", "budget_hit", "sqp", "pcg", "zero_pcg_rows",
                "admm", "admm_linsys", "admm_linsys_mode", "pred_err", "limits_mode", "linsys_mode", *DIAGNOSTICS)


@dataclass(frozen=True)
class FixedReach:
    """Hold one reaching goal for the full window, including after independently reported arrival."""
    reach: Reach
    duration_s: float = DURATION_S

    def at(self, t: float, n: int, dt: float):
        return self.reach.at(t, n, dt)

    def done(self, state: tm.State, t: float) -> bool:
        return t >= self.duration_s - 1e-9


def task_configuration(name: str, n: int = 128):
    """Construct the matched task pair with a cost menu of exactly n knots."""
    if n not in KNOTS:
        raise ValueError(f"knots must be one of {KNOTS}")
    if name == "figure8":
        task = Figure8(period_s=DURATION_S)
        plant = tm.SimConfig(friction=tm.urdf_friction(), integrator="euler", substep=.001)
        base, cost = FIGURE8, tm.default_cost(n, velocity=.01, torque=1e-6)
    elif name == "reach":
        posture = tm.Q_START + 3 * (tm.Q_REACH - tm.Q_START)
        limits = tm.limits()
        if np.any(posture < limits.lower) or np.any(posture > limits.upper):
            raise ValueError("the frozen medium-reach target posture exceeds the raw joint limits")
        task = FixedReach(Reach(target_m=tuple(map(float, tm.ee_position(posture)))))
        plant = tm.SimConfig(friction=tm.urdf_friction(), integrator="euler", substep=.001)
        base, cost = tm.TIAGO, tm.default_cost(n)
    else:
        raise ValueError(f"unknown task {name}")
    params = replace(base, N=n, dt_s=.01, cost=cost, max_sqp_time_s=None, warmup=True,
                     solver=replace(base.solver, max_sqp_iters=1),
                     warm_start="hold", friction=None, reseed_threshold=None,
                     linsys="pcg", reset_rho_each_step=True,
                     admm_linsys=None, collision=None)
    return task, plant, params


def case_order(knots: tuple[int, ...], repeats: int) -> list[tuple[int, str, str, int]]:
    """Alternate solver order across fresh-controller repeats at each knot count."""
    return [(n, task, solver, repeat) for n in knots for repeat in range(1, repeats + 1)
            for task in TASKS for solver in (SOLVERS if repeat % 2 else SOLVERS[::-1])]


def fingerprint(task, plant: tm.SimConfig, params: tm.MpcParams) -> str:
    """Hash equal scientific inputs after removing only the intended solver difference."""
    params = replace(params, linsys=None, solver=replace(params.solver, linsys=None))
    config = tm.to_dict(dict(task=task, plant=plant, params=params, barrier=BARRIER, batch_size=1,
                             settings=SETTINGS, law=tm.SampleKnot(), compensation=None))
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(tm.to_dict(value), indent=2, allow_nan=False) + "\n")


def command_text(args: list[str]) -> str:
    """Read available provenance without making optional device tools launch gates."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return "command unavailable"
    return result.stdout + result.stderr


class Quality:
    """Accumulate quality scalars online without retaining states, commands or plans."""

    def __init__(self, task: Figure8 | FixedReach):
        self.task, self.limits = task, tm.limits()
        self.count, self.error_squared, self.max_error = 0, 0., 0.
        self.final_distance, self.final_speed, self.arrival_s = None, None, None
        self.excess = dict(position_rad=0., speed_rad_s=0., torque_nm=0.)
        self.clipped_commands = 0

    def __call__(self, tick: tm.Tick) -> None:
        state = tick.state
        t = state.t
        target = (self.task.path(np.array([t]))[0] if isinstance(self.task, Figure8)
                  else self.task.reach.target_m)
        error = float(np.linalg.norm(tm.ee_position(state.q) - target))
        self.count += 1
        self.error_squared += error * error
        self.max_error = max(self.max_error, error)
        self.final_distance = error
        self.excess["position_rad"] = max(self.excess["position_rad"],
                                           float(np.max(self.limits.lower - state.q)),
                                           float(np.max(state.q - self.limits.upper)))
        self.excess["speed_rad_s"] = max(self.excess["speed_rad_s"],
                                          float(np.max(np.abs(state.qd) - self.limits.velocity)))
        if tick.command is not None:
            self.excess["torque_nm"] = max(self.excess["torque_nm"],
                                            float(np.max(np.abs(tick.command) - self.limits.effort)))
        self.clipped_commands += int(tick.clipped)
        if isinstance(self.task, FixedReach):
            self.final_speed = float(np.linalg.norm(tm.ee_velocity(state.q, state.qd)))
            if (self.arrival_s is None and error <= self.task.reach.position_tol_m
                    and self.final_speed <= self.task.reach.speed_tol_m_s):
                self.arrival_s = t

    def result(self) -> dict:
        return dict(observations=self.count, rmse_m=None if not self.count else (self.error_squared / self.count) ** .5,
                    max_error_m=self.max_error, final_distance_m=self.final_distance,
                    final_speed_m_s=self.final_speed, arrival_s=self.arrival_s,
                    limits_pass=None if not self.count else max(self.excess.values()) <= 1e-5, max_limit_excess=self.excess,
                    clipped_commands=self.clipped_commands)


def save_stats(path: Path, rows: list[dict]) -> None:
    """Write scalar solver statistics only, including partial data from failed cases."""
    columns = ("simulation_s", *SCALAR_STATS)
    np.savez_compressed(path, **{key: np.asarray([row[key] for row in rows]) for key in columns})
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def install_barriers(mpc: tm.Mpc) -> dict:
    """Replace the wrapper's ADMM rows before warm-up, preserving its planning bounds."""
    solver = mpc.controller.solver
    previous = solver.get_row_groups()
    bounds = [(group["kind"], np.array(group["lo"], copy=True), np.array(group["hi"], copy=True))
              for group in previous]
    if len(bounds) != 3:
        raise ValueError("the barrier experiment requires only the three canonical limit groups")
    solver.enable_limit_barrier(**BARRIER)
    current = solver.get_row_groups()
    if [group["kind"] for group in current] != [kind for kind, _, _ in bounds]:
        raise RuntimeError("switching to barriers changed the canonical limit groups")
    for index, (_, lower, upper) in enumerate(bounds):
        solver.set_row_group_bounds(index, lower, upper)
    mpc.controller.limits = solver.row_mech
    if solver.admm_active or mpc.controller.limits != "barrier":
        raise RuntimeError("barrier installation left ADMM active or selected another mechanism")
    return dict(mechanism=solver.row_mech, admm_active=solver.admm_active,
                barrier=BARRIER, row_groups=solver.get_row_groups())


def scalar_diagnostics(mpc: tm.Mpc) -> dict:
    """Read existing CUDA-event stage spans and batch identity without retaining trajectories."""
    raw = mpc.step.solve.stats
    stages = np.asarray(raw.stage_times_us, dtype=float)
    if stages.ndim != 2 or stages.shape[1] != len(STAGE_NAMES) or not np.isfinite(stages).all():
        raise ValueError("missing or invalid native stage timings")
    steps = np.asarray(raw.step_size)
    return dict(zip(STAGE_NAMES, map(float, stages.sum(axis=0))),
                initial_merit_us=float(raw.initial_merit_time_us),
                linear_solve_us=float(np.sum(raw.pcg_times_us)),
                step_size=float(steps[-1, 0]) if steps.size else 0., line_search_ran=bool(steps.size),
                completed_unix_s=time.time(), batch_size=int(mpc.controller.solver.batch_size))


def run_case(
    out: Path,
    n: int,
    name: str,
    mode: str,
    repeat: int,
    binary: Path | None,
    build_error: str | None = None,
) -> dict:
    """Run one simulator episode and persist only solve statistics and aggregate quality."""
    out.mkdir(parents=True, exist_ok=False)
    task, plant_config, params = task_configuration(name, n)
    params = replace(params, linsys=mode, solver=replace(params.solver, linsys=mode))
    rows: list[dict] = []
    quality, diagnostics = Quality(task), Counter()
    mpc, plant = None, None
    config = dict(task=task, plant=plant_config, params=params, settings=SETTINGS,
                  law=tm.SampleKnot(), compensation=None, barrier=BARRIER, batch_size=1,
                  input_fingerprint=fingerprint(task, plant_config, params))
    case = dict(knots=n, task=name, solver=mode, repeat=repeat, config=config,
                status="running", reason=None, end_t=None, started_unix_s=time.time(),
                error=None, binary=None, binary_sha256=None)
    write_json(out / "case.json", case)
    try:
        if build_error is not None:
            raise RuntimeError(f"the prior N{n} build failed; no repeated build attempted:\n{build_error}")
        binary = tm.compile_binary(params, jobs=1) if binary is None else binary
        case.update(binary=str(binary), binary_sha256=sha256(binary))
        mpc = tm.Mpc(params, binary)
        config["batch_size"] = int(mpc.controller.solver.batch_size)
        if config["batch_size"] != 1:
            raise ValueError("the PCG/BDSV comparison requires batch size one")
        mpc.controller.solver.set_collect_stats(True)
        config["native_limits"] = install_barriers(mpc)
        config["params"] = mpc.params
        write_json(out / "case.json", case)
        plant = plant_config.build()

        def completed(item: tm.SolveCompleted) -> None:
            stats = dict(item.plan.stats, limits_mode=mpc.step.limits,
                         linsys_mode=mpc.controller.solver.params.linsys, **scalar_diagnostics(mpc))
            if set(SCALAR_STATS) - stats.keys():
                raise ValueError("the solver omitted required scalar timing/work statistics")
            row = dict(simulation_s=float(item.request_time_s), **{key: stats[key] for key in SCALAR_STATS})
            if any(np.ndim(value) != 0 for value in row.values()):
                raise ValueError("a recorded solver statistic is not scalar")
            rows.append(row)

        def diagnostic(item: tm.Diagnostic) -> None:
            diagnostics[item.kind] += 1

        result = tm.run_virtual_time(plant, mpc, task, tm.SampleKnot(), settings=SETTINGS,
                                     compensation=None, duration=DURATION_S, tick_callbacks=[quality],
                                     solve_completed_callbacks=[completed], diagnostic_callbacks=[diagnostic])
        case.update(status=result.status, reason=result.reason, end_t=result.t)
    except KeyboardInterrupt:
        case.update(status="interrupted", reason="KeyboardInterrupt", error=traceback.format_exc())
    except Exception:
        case.update(status="error", error=traceback.format_exc())
    finally:
        if plant is not None:
            plant.close()
        if mpc is not None:
            mpc.close()
        save_stats(out / "stats.npz", rows)
        case.update(quality=quality.result(), diagnostic_counts=dict(diagnostics),
                    solves=len(rows), ended_unix_s=time.time())
        write_json(out / "case.json", case)
    return case


def run_study(out: Path, knots: tuple[int, ...], repeats: int) -> bool:
    """Run a serial stats-only sweep with pinned inputs and continuously sampled GPU clocks."""
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    source.mkdir()
    root = Path(__file__).resolve().parents[1]
    for relative in ("experiments/pcg_bdsv_histograms.py", "pyproject.toml", "uv.lock",
                     "experiments/figure8.py", "experiments/reach.py", "tiago_mpc/mpc.py", "tiago_mpc/loops.py",
                     "tiago_mpc/laws.py", "tiago_mpc/sim.py"):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / relative).read_bytes())
    (source / "tracked.diff").write_text(command_text(["git", "-C", str(root), "diff", "HEAD"]))
    provenance = tm.provenance()
    provenance.update(machine=platform.uname()._asdict(), command=sys.argv,
                      gpu_before=command_text(["nvidia-smi"]),
                      gpu_name=command_text(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).strip(),
                      nvcc=command_text(["nvcc", "--version"]),
                      jetson_power_mode=command_text(["nvpmodel", "-q"]) if platform.machine() == "aarch64" else None,
                      jetson_clock_state=command_text(["jetson_clocks", "--show"]) if platform.machine() == "aarch64" else None)
    write_json(out / "provenance.json", provenance)
    manifest = dict(protocol=PROTOCOL, batch_size=1, duration_s=DURATION_S, tasks=TASKS, knots=knots, repeats=repeats,
                    settings=SETTINGS, storage="scalar solve stats and aggregate quality only", cases=[])
    write_json(out / "study.json", manifest)
    monitor = None
    binaries, build_errors = {}, {}
    with (out / "gpu-clocks.csv").open("w") as clocks:
        try:
            try:
                smi = shutil.which("nvidia-smi")
                usable_smi = smi and subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"],
                                                    capture_output=True, check=False).returncode == 0
                if usable_smi:
                    manifest["clock_monitor"] = "nvidia-smi"
                    monitor = subprocess.Popen([
                        smi, "--query-gpu=timestamp,uuid,clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw,utilization.gpu",
                        "--format=csv", "--loop-ms=200"], stdout=clocks, stderr=subprocess.STDOUT)
                elif shutil.which("tegrastats"):
                    clocks.write("nvidia-smi unavailable; read tegrastats.log for Jetson frequency, load and thermal samples\n")
                    manifest["clock_monitor"] = "tegrastats"
                    with (out / "tegrastats.log").open("w") as tegra:
                        monitor = subprocess.Popen(["tegrastats", "--interval", "200"],
                                                   stdout=tegra, stderr=subprocess.STDOUT)
                else:
                    manifest["clock_monitor"] = "unavailable"
                    clocks.write("nvidia-smi and tegrastats unavailable; no continuous GPU clock measurement\n")
            except FileNotFoundError:
                manifest["clock_monitor"] = "unavailable"
                clocks.write("nvidia-smi unavailable; no continuous GPU clock measurement\n")
            for n, name, mode, repeat in case_order(knots, repeats):
                relative = Path("cases") / f"N{n}" / name / f"repeat-{repeat:02d}-{mode}"
                print(f"Starting {relative}", flush=True)
                case = run_case(out / relative, n, name, mode, repeat, binaries.get(n), build_errors.get(n))
                if case["binary"] is not None:
                    binaries[n] = Path(case["binary"])
                elif case["status"] == "error":
                    build_errors[n] = case["error"]
                manifest["cases"].append(dict(knots=n, task=name, solver=mode, repeat=repeat,
                                               path=str(relative), status=case["status"]))
                write_json(out / "study.json", manifest)
                print(f"Finished {relative}: {case['status']}; {case['solves']} solves", flush=True)
                if case["status"] == "interrupted":
                    break
        finally:
            if monitor is not None:
                monitor.terminate()
                monitor.wait(timeout=10)
    (out / "gpu-after.txt").write_text(command_text(["nvidia-smi"]))
    return analyze(out)


def timing_rows(out: Path) -> tuple[list[dict], list[dict]]:
    """Validate actual paired configs and read scalar statistics without dropping failed cases."""
    study = json.loads((out / "study.json").read_text())
    if study["protocol"] != PROTOCOL:
        raise ValueError("analyze this study with the source saved beside its measurements")
    rows, quality, resolved = [], [], {}
    for entry in study["cases"]:
        directory = out / entry["path"]
        case = json.loads((directory / "case.json").read_text())
        stats = np.load(directory / "stats.npz", allow_pickle=False)
        count = len(stats["simulation_s"])
        for key in SCALAR_STATS:
            if key not in stats or len(stats[key]) != count or stats[key].ndim != 1:
                raise ValueError(f"{directory}: missing or nonscalar {key}")
        for key in ("wall_s", "solve_time_us"):
            if not np.isfinite(stats[key]).all() or np.any(stats[key] <= 0):
                raise ValueError(f"{directory}: invalid timing in {key}")
        if count:
            if (not np.all(stats["linsys_mode"] == entry["solver"])
                    or not np.all(stats["batch_size"] == 1) or case["config"]["batch_size"] != 1
                    or not np.all(stats["limits_mode"] == "barrier")
                    or not np.all(stats["admm_linsys_mode"] == "")
                    or np.any(stats["admm"] != 0) or np.any(stats["admm_linsys"] != 0)):
                raise ValueError(f"{directory}: actual linear solver differs from request")
            actual = case["config"]["params"]
            if (actual["N"] != entry["knots"] or actual["dt_s"] != .01 or actual["max_sqp_time_s"] is not None
                    or actual["solver"]["max_sqp_iters"] != 1 or actual["solver"]["max_pcg_iters"] != 120
                    or actual["linsys"] != entry["solver"] or actual["solver"]["linsys"] != entry["solver"]
                    or actual["warm_start"] != "hold" or actual["friction"] is not None
                    or actual["reseed_threshold"] is not None):
                raise ValueError(f"{directory}: resolved settings differ from the frozen protocol")
            intended = task_configuration(entry["task"], entry["knots"])
            if case["config"]["input_fingerprint"] != fingerprint(*intended):
                raise ValueError(f"{directory}: scientific input fingerprint differs")
            if (case["config"]["plant"] != tm.to_dict(intended[1])
                    or case["config"]["law"] != tm.to_dict(tm.SampleKnot())
                    or case["config"]["settings"] != tm.to_dict(SETTINGS)
                    or case["config"]["barrier"] != BARRIER
                    or case["config"]["native_limits"]["mechanism"] != "barrier"
                    or case["config"]["native_limits"]["admm_active"]
                    or case["config"]["compensation"] is not None):
                raise ValueError(f"{directory}: plant, direct-effort law or timing differs from the frozen protocol")
            paired = dict(case["config"], params=dict(actual, linsys=None,
                          solver=dict(actual["solver"], linsys=None)))
            previous = resolved.setdefault((entry["knots"], entry["task"]), paired)
            if paired != previous:
                raise ValueError(f"{directory}: actual paired configs differ beyond linear solver")
        for i in range(count):
            rows.append(dict(knots=entry["knots"], task=entry["task"], solver=entry["solver"], repeat=entry["repeat"],
                             update=i, simulation_s=float(stats["simulation_s"][i]), first_update=i == 0,
                             wall_ms=float(stats["wall_s"][i]) * 1000, native_ms=float(stats["solve_time_us"][i]) / 1000,
                             pred_err=float(stats["pred_err"][i]),
                             **{key: float(stats[key][i]) for key in DIAGNOSTICS},
                             **{key: int(stats[key][i]) for key in ("sqp", "pcg", "admm", "admm_linsys")}))
        expected = int(round(DURATION_S / SETTINGS.update_period))
        complete = (count == expected and case["status"] in ("done", "timeout")
                    and case["end_t"] is not None and abs(case["end_t"] - DURATION_S) < 1e-6)
        quality.append(dict(knots=entry["knots"], task=entry["task"], solver=entry["solver"], repeat=entry["repeat"],
                            status=case["status"], error=case["error"], solves=count, expected_solves=expected,
                            recording_complete=complete,
                            unexpected_sqp_count=int(np.count_nonzero(stats["sqp"] != 1)),
                            unexpected_admm_count=int(np.count_nonzero((stats["admm"] != 0)
                                                                      | (stats["admm_linsys"] != 0))),
                            pcg_cap_hits=int(np.count_nonzero(stats["pcg"] >= 120)) if entry["solver"] == "pcg" else 0,
                            **case["quality"]))
    return rows, quality


def distribution(values: np.ndarray) -> dict:
    """Describe serial per-update spread, not uncertainty across independent repeats."""
    return dict(n=len(values), mean_ms=float(np.mean(values)), median_ms=float(np.median(values)),
                sd_ms=float(np.std(values)), variance_ms2=float(np.var(values)),
                p05_ms=float(np.percentile(values, 5)), p95_ms=float(np.percentile(values, 95)),
                p99_ms=float(np.percentile(values, 99)), min_ms=float(np.min(values)), max_ms=float(np.max(values)))


def common_bins(values: np.ndarray, multiplier: float = 1.) -> np.ndarray:
    """Use pooled bin edges for both solvers while retaining the complete measured tail."""
    count = max(20, min(160, len(np.histogram_bin_edges(values, bins="fd")) - 1))
    count = max(10, int(round(count * multiplier)))
    low, high = float(values.min()), float(values.max())
    if low == high:
        high = low + max(1e-6, low * 1e-3)
    return np.linspace(low, high, count + 1)


def analyze(out: Path) -> bool:
    """Write per-knot histograms and summaries from stats-only records, including partial failures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    study = json.loads((out / "study.json").read_text())
    rows, quality = timing_rows(out)
    analysis_source = out / "analysis-source"
    analysis_source.mkdir(exist_ok=True)
    analysis_file = analysis_source / "pcg_bdsv_histograms.py"
    analysis_file.write_bytes(Path(__file__).read_bytes())
    write_json(out / "analysis-provenance.json", dict(script_sha256=sha256(analysis_file), measurements_changed=False))
    provenance = json.loads((out / "provenance.json").read_text())
    gpu_name = provenance.get("gpu_name")
    if not gpu_name or "unavailable" in gpu_name:
        gpu_name = "Jetson" if provenance["machine"]["machine"] == "aarch64" else provenance["machine"]["node"]
    with (out / "solve-times.csv").open("w", newline="") as stream:
        columns = list(rows[0]) if rows else ["knots", "task", "solver", "repeat", "update", "simulation_s", "wall_ms", "native_ms"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    attempted = {(entry["knots"], entry["task"], entry["solver"], entry["repeat"]) for entry in study["cases"]}
    planned = set(case_order(tuple(study["knots"]), study["repeats"]))
    completion = dict(planned_cases=len(planned), attempted_cases=len(attempted), missing_cases=sorted(planned - attempted),
                      unexpected_cases=sorted(attempted - planned), complete=(attempted == planned
                      and len(study["cases"]) == len(planned) and all(q["recording_complete"] for q in quality)))
    write_json(out / "completion.json", completion)
    write_json(out / "quality.json", quality)
    stats, work = [], []
    colors = dict(pcg="#1765ad", bdsv="#df7417")
    for n in study["knots"]:
        selected_n = [row for row in rows if row["knots"] == n]
        if not selected_n:
            continue
        for task in TASKS:
            for solver in SOLVERS:
                selected = [row for row in selected_n if row["task"] == task and row["solver"] == solver]
                if not selected:
                    continue
                for repeat in [None, *sorted({row["repeat"] for row in selected})]:
                    subset = selected if repeat is None else [row for row in selected if row["repeat"] == repeat]
                    for boundary in ("wall_ms", "native_ms"):
                        stats.append(dict(knots=n, task=task, solver=solver, repeat=repeat, boundary=boundary,
                                          **distribution(np.array([row[boundary] for row in subset]))))
                work.append(dict(knots=n, task=task, solver=solver, **{
                    key: dict(min=min(row[key] for row in selected), median=float(np.median([row[key] for row in selected])),
                              max=max(row[key] for row in selected)) for key in ("sqp", "pcg", "admm", "admm_linsys")}))
        for multiplier, suffix in ((.5, "-coarse"), (1., ""), (2., "-fine")):
            fig, axes = plt.subplots(2, len(TASKS), figsize=(6 * len(TASKS), 7.5),
                                     constrained_layout=True, squeeze=False)
            for col, task in enumerate(TASKS):
                for r, boundary in enumerate(("wall_ms", "native_ms")):
                    ax = axes[r, col]
                    selected = [row for row in selected_n if row["task"] == task]
                    if not selected:
                        ax.set_title(f"{task}: no returned timings")
                        continue
                    bins = common_bins(np.array([row[boundary] for row in selected]), multiplier)
                    for solver in SOLVERS:
                        data = np.array([row[boundary] for row in selected if row["solver"] == solver])
                        if len(data):
                            ax.hist(data, bins=bins, weights=np.full(len(data), 100 / len(data)), histtype="step",
                                    linewidth=1.7, label=f"{'PCG' if solver == 'pcg' else 'BDSV'} (n={len(data)})", color=colors[solver])
                    ax.set(title=f"{task} — {'controller step' if boundary == 'wall_ms' else 'native solver'}",
                           xlabel="Solve time (ms)", ylabel="Solves per bin (%)")
                    ax.grid(alpha=.2)
                    ax.legend(fontsize=9)
            fig.suptitle(f"{gpu_name} · N{n} · SQP1 · {DURATION_S:g} s figure-eight\n"
                         "Limit barriers · 20 Hz updates · ADMM off\n"
                         "Direct torque · unshifted warm starts", fontsize=11)
            fig.savefig(out / f"histograms-N{n}{suffix}.png", dpi=170)
            fig.savefig(out / f"histograms-N{n}{suffix}.pdf")
            plt.close(fig)
        fig, axes = plt.subplots(2, len(TASKS), figsize=(6 * len(TASKS), 7.5),
                                 constrained_layout=True, squeeze=False)
        for col, task in enumerate(TASKS):
            for solver in SOLVERS:
                selected = [row for row in selected_n if row["task"] == task and row["solver"] == solver]
                if not selected:
                    continue
                label = "PCG" if solver == "pcg" else "BDSV"
                axes[0, col].scatter([row["simulation_s"] for row in selected],
                                     [row["wall_ms"] for row in selected],
                                     s=5, alpha=.35, color=colors[solver], label=label)
                axes[1, col].scatter([row["pred_err"] for row in selected],
                                     [row["wall_ms"] for row in selected],
                                     s=5, alpha=.35, color=colors[solver], label=label)
            axes[0, col].set(title=task, xlabel="Simulation time (s)", ylabel="Controller step (ms)")
            axes[1, col].set(xlabel="GATO warm-start error (state norm)", ylabel="Controller step (ms)")
            for ax in axes[:, col]:
                ax.grid(alpha=.2)
                ax.legend()
        fig.suptitle(f"{gpu_name} · N{n} · {DURATION_S:g} s figure-eight\n"
                     "Timing and unshifted warm-start error", fontsize=11)
        fig.savefig(out / f"timing-diagnostics-N{n}.png", dpi=170)
        fig.savefig(out / f"timing-diagnostics-N{n}.pdf")
        plt.close(fig)
    write_json(out / "distributions.json", stats)
    write_json(out / "work-summary.json", work)
    report = [f"# One-SQP PCG versus BDSV stats-only sweep — {gpu_name}", "",
              f"Study complete: **{completion['complete']}**; {len(attempted)}/{len(planned)} planned cases recorded. "
              "Missing/error/partial cases remain explicit in completion.json and quality.json; no failed cases are discarded.", "",
              f"Each case executes one whole {DURATION_S:g}-second figure-eight, "
              "with 500 Hz commands, 20 Hz control updates, 10 ms knots, "
              "zero declared solve latency and no latency compensation. N changes prediction horizon to (N−1)*10 ms. "
              "Costs are reconstructed for N with unchanged per-knot weights and terminal weights. "
              "Each fresh controller has a one-SQP-iteration cap, PCG cap120 and no wall-time cap. "
              "ADMM is disabled. Relaxed limit barriers use mu=0.003 and delta=0.05 with unchanged planning bounds. "
              "Both plants use the fitted URDF friction with coupled StickSlip and 1 ms Euler substeps. "
              "The predictor is loss-free. SampleKnot applies plan torques without PD or friction feedforward; "
              "the usual simulated effort clipping remains active and clipped-command counts are reported. "
              "warm_start='hold' reuses the previous winning trajectory without shifting; automatic reseeding is off. "
              "Only the plain Schur solver (linsys='pcg' versus 'bdsv') differs within a task/N pair. Warm-up and compilation are excluded; "
              "Batch size is exactly one at every horizon and checked in the native solver and every timing sample. "
              "all returned updates including the first are retained. No trajectories, ticks, plans or replays are saved.", "",
              "Controller step wall time includes staging, native solve and unshifted warm-start retention; it excludes Python "
              "reference/cost-menu construction, physics integration and statistics callbacks. Native time is "
              "solve_time_us. Serial-call SD/variance describe observed spread, not repeat uncertainty. "
              "distributions.json reports pooled and individual repeats. Histograms share bin edges between solvers; "
              "coarse/fine variants retain every measured tail. work-summary.json reports PCG work; BDSV's native "
              "iteration marker is not an equivalent work unit. quality.json retains PCG cap hits and zero ADMM work. "
              "A finite plan or iteration-cap exit does not establish convergence.", "",
              "Per-update stats include initial merit, KKT, Schur, linear solve, merit and line-search event spans, "
              "accepted step and completion wall timestamp. CUDA-event spans can contain runtime scheduling delays "
              "and are not exclusive arithmetic timings. Clock/load monitoring uses nvidia-smi, or tegrastats on "
              "Jetson when nvidia-smi is unavailable; the log records unavailable tools. Power mode and clocks are "
              "queried read-only when available, never changed. A timing peak is not by itself an algorithmic mode.", "",
              "| N | Task | Solver | Calls | Median wall ms | SD wall ms | p95 wall ms | Median native ms |",
              "|---:|---|---|---:|---:|---:|---:|---:|"]
    for item in stats:
        if item["repeat"] is None and item["boundary"] == "wall_ms":
            native = next(s for s in stats if s["knots"] == item["knots"] and s["task"] == item["task"]
                          and s["solver"] == item["solver"] and s["repeat"] is None and s["boundary"] == "native_ms")
            report.append(f"| {item['knots']} | {item['task']} | {item['solver']} | {item['n']} | "
                          f"{item['median_ms']:.4f} | {item['sd_ms']:.4f} | {item['p95_ms']:.4f} | {native['median_ms']:.4f} |")
    for n in study["knots"]:
        if any(row["knots"] == n for row in rows):
            report.extend(["", f"![N{n} solve-time histograms](histograms-N{n}.png)", "",
                           f"![N{n} timing and warm-start diagnostics](timing-diagnostics-N{n}.png)"])
    report.extend(["", "| N | Task | Solver | Repeat | Status | Solves | RMS mm | Final error mm | Arrival s | Limits pass | Clipped commands |",
                   "|---:|---|---|---:|---|---:|---:|---:|---:|---|---:|"])
    for item in quality:
        def metric(key, scale=1.):
            return "—" if item.get(key) is None else f"{item[key] * scale:.3f}"
        report.append(f"| {item['knots']} | {item['task']} | {item['solver']} | {item['repeat']} | "
                      f"{item['status']} | {item['solves']} | {metric('rmse_m', 1000)} | "
                      f"{metric('final_distance_m', 1000)} | {metric('arrival_s')} | "
                      f"{item.get('limits_pass')} | {item.get('clipped_commands', 0)} |")
    report.extend(["", "Closed-loop states can differ with numerical solver outputs; this is a matched-policy "
                   "comparison, not an identical-KKT replay. Arrival is reported independently without shortening "
                   "the reach window. Aggregate raw-limit checks are sampled command-tick observations, not continuous "
                   "safety certification. gpu-clocks.csv records available clocks during measurement; unavailable "
                   "tools are stated explicitly and no power modes or clocks are changed.", "",
                   "Regenerate with `python -m experiments.pcg_bdsv_histograms --analyze PATH`. "
                   "Run with `python -m experiments.pcg_bdsv_histograms --output _expr_data/pcg-bdsv-mismatch "
                   "--knots 128 --repeats 3` under the workspace heavy.lock. "
                   "Prediction error is GATO's state-space norm against its retained warm start; it is not "
                   "a pure friction-error measurement when shifting is disabled. Multiple timing peaks are "
                   "a hypothesis, not an acceptance criterion."])
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    return completion["complete"]


def describe(knots: tuple[int, ...], repeats: int) -> dict:
    """Describe the frozen sweep without loading a compiled module or touching the GPU."""
    tasks = []
    for n in knots:
        for name in TASKS:
            task, plant, params = task_configuration(name, n)
            tasks.append(dict(knots=n, task=name, prediction_horizon_s=(n - 1) * params.dt_s,
                              dt_s=params.dt_s, max_sqp_iters=params.solver.max_sqp_iters,
                              max_pcg_iters=params.solver.max_pcg_iters, limits="barrier", barrier=BARRIER, admm_active=False,
                              plant=plant, predictor_friction=params.friction, warm_start=params.warm_start,
                              reseed_threshold=params.reseed_threshold, law=tm.SampleKnot(),
                              max_sqp_time_s=params.max_sqp_time_s, fingerprint=fingerprint(task, plant, params)))
    return dict(protocol=PROTOCOL, batch_size=1, knots=knots, repeats=repeats, solvers=SOLVERS, task_names=TASKS,
                duration_s=DURATION_S, settings=SETTINGS, cases=len(case_order(knots, repeats)),
                expected_solves_per_case=int(round(DURATION_S / SETTINGS.update_period)),
                expected_total_solves=len(case_order(knots, repeats)) * int(round(DURATION_S / SETTINGS.update_period)),
                storage="scalar solve statistics and aggregate quality only; no trajectories/ticks/plans/replays", tasks=tasks)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--analyze", type=Path, help="regenerate statistics and plots without GPU work")
    parser.add_argument("--describe", action="store_true", help="print the study plan without compiled modules or GPU work")
    parser.add_argument("--knots", nargs="+", type=int, choices=KNOTS, default=list(KNOTS))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if sum((bool(args.output), bool(args.analyze), args.describe)) != 1:
        parser.error("choose exactly one of --output, --analyze and --describe")
    if args.repeats < 1 or len(set(args.knots)) != len(args.knots):
        parser.error("--repeats must be positive and --knots must not contain duplicates")
    handle_stop_signals()
    if args.describe:
        print(json.dumps(tm.to_dict(describe(tuple(args.knots), args.repeats)), indent=2))
        return 0
    if args.analyze:
        complete = analyze(args.analyze)
    else:
        complete = run_study(args.output, tuple(args.knots), args.repeats)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
