"""Six execution controllers on a frictional arm with armature and sampled feedback.

Run as python -m experiments.friction_compensation --dry-run, then --output
_expr_data/friction-compensation/<fresh-name>. Defaults: 30 ms knots, terminal
EE weight 80, mu=1, qd=.01, u=1e-4, ADMM rho_u=5e-4. Execution controllers and
commands and feedback run every 2 ms; TO updates every 20 ms. Override
--feedback-period .02 for the separate slow-feedback comparison.
The declared 5.5 ms solve delay is a recorded-run approximation, not transport
calibration. Plant truth is recorded separately and used for motion metrics.
"""
from dataclasses import dataclass, replace
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

from experiments.common import handle_stop_signals, limit_excess, parse_solve_latency, run_experiment
from experiments.figure8 import FIGURE8, Figure8, evaluate, metrics
from experiments.replay import save_replay
import tiago_mpc as tm
from tiago_mpc.tiago import ARMATURE_KG_M2

FEEDBACK_PERIOD_S = .002
MPC_FRICTION_WIDTH_RAD_S = .2  # 30 ms knots: near-zero damping screen, not a fitted transition


@dataclass(frozen=True)
class SampledSimConfig:
    """Experiment-local observation sampling; transport and torque delivery are immediate."""
    physical: tm.SimConfig
    feedback_period_s: float = FEEDBACK_PERIOD_S

    def __post_init__(self):
        if not np.isfinite(self.feedback_period_s) or self.feedback_period_s <= 0:
            raise ValueError('feedback_period_s must be finite and positive')

    def build(self):
        return SampledSim(self)


class SampledSim:
    """The existing plant with held observations and separately recorded true states.

    Sensor diagnostics occur at their own times, also between command ticks. This wrapper
    supports virtual time only. No observation noise or additional transport delay.
    """
    def __init__(self, config):
        self.config, self.physical = config, config.physical.build()
        self.run_state = self.physical.run_state
        self._diagnostics = []

    def start(self, clock):
        if not isinstance(clock, tm.VirtualClock):
            raise ValueError('SampledSim supports virtual time only')
        self.clock = clock
        self.physical.start(clock)
        self._start, self._sample_index = clock.now(), 1
        self._sample = self.physical.state()
        self._last_truth_t = None
        self._diagnostics.clear()

    def state(self):
        truth = self.physical.state()
        if truth.t != self._last_truth_t:
            self._diagnostics.append(tm.Diagnostic(truth.t, 'plant_state', 'True simulated state recorded separately',
                                                 dict(q=truth.q.tolist(), qd=truth.qd.tolist())))
            self._last_truth_t = truth.t
        return tm.State(self._sample.t, self._sample.q.copy(), self._sample.qd.copy())

    def advance(self, t):
        while True:
            sample_t = self._start + self._sample_index * self.config.feedback_period_s
            if sample_t > t + 1e-9:  # match the loop's scheduling tolerance over a full cycle
                break
            self.physical.advance(min(sample_t, t))  # clamp rounding at a shared sample/tick boundary
            self._sample = self.physical.state()
            self._sample_index += 1
        self.physical.advance(t)

    def apply(self, command):
        self.physical.apply(command)

    def take_diagnostics(self):
        diagnostics, self._diagnostics = self._diagnostics, []
        return diagnostics + self.physical.take_diagnostics()

    def stop(self):
        self.physical.stop()

    def close(self):
        self.physical.close()


PARAMS = replace(FIGURE8, dt_s=.03, friction=None, armature=ARMATURE_KG_M2,
                 solver=replace(FIGURE8.solver, mu=1.),
                 cost=tm.default_cost(FIGURE8.N, velocity=.01, torque=1e-4, terminal_position=80.),
                 admm=replace(FIGURE8.admm, rho_u=5e-4))
TIMING = tm.VirtualTimeSettings(command_period=.002, update_period=.02, solve_latency_s=.0055)
PLANT_COULOMB_J1_4_NM = (2.01, 2.39, 2.03, 1.91)
PLANT_VISCOUS_J1_4_NM_S_RAD = (4.44, 5.07, 2.87, 2.82)
LABELS = {'default': 'MPC only', 'feedforward': 'Friction feedforward',
          'pd': 'PD feedback', 'both': 'Feedforward + PD',
          'mpc_friction': 'Friction in MPC', 'mpc_friction_pd': 'Friction in MPC + PD'}


def plant_config(feedback_period_s=FEEDBACK_PERIOD_S):
    nominal = tm.urdf_friction()
    friction = replace(nominal, coulomb_nm=PLANT_COULOMB_J1_4_NM + nominal.coulomb_nm[4:],
                       viscous_nm_s_per_rad=PLANT_VISCOUS_J1_4_NM_S_RAD + nominal.viscous_nm_s_per_rad[4:])
    return SampledSimConfig(tm.SimConfig(substep=.001, friction=friction, integrator='euler',
                                        armature=ARMATURE_KG_M2), feedback_period_s)


def cases():
    estimate, pd, width = tm.urdf_friction(), tm.PlanFeedback(), tm.Tanh(.03)  # the study's, not the defaults
    return {'default': tm.SampleKnot(), 'feedforward': tm.FrictionFeedforward(estimate, tm.SampleKnot(), width),
            'pd': pd, 'both': tm.FrictionFeedforward(estimate, pd, width),
            'mpc_friction': tm.SampleKnot(), 'mpc_friction_pd': pd}


def case_params(case, params=PARAMS, width=MPC_FRICTION_WIDTH_RAD_S):
    """Internal friction replaces external feedforward; every predictor has armature."""
    return replace(params, friction=tm.urdf_friction() if case.startswith('mpc_friction') else None,
                   friction_law=tm.Tanh(width))


def configuration(params=PARAMS, plant=None, timing=TIMING, width=MPC_FRICTION_WIDTH_RAD_S):
    delay = ('This desktop GPU\'s measured GATO step wall time; not recorded Jetson latency. '
             if timing.solve_latency_s == 'measured'
             else f'Declared solve delay {timing.solve_latency_s:g}s, activation at a command tick. ')
    if timing.solve_latency_s == .0055:
        delay += 'Rounded from the recorded native solve mean (5.4441ms), not end-to-end latency. '
    return dict(provenance=tm.provenance(), urdf_sha256=hashlib.sha256(tm.FULL_URDF.read_bytes()).hexdigest(),
                params=tm.to_dict(params), task=tm.to_dict(Figure8()),
                plant=tm.to_dict(plant if plant is not None else plant_config()), settings=tm.to_dict(timing),
                loop="run_virtual_time", compensation=None, safety=tm.to_dict(tm.ExecutionSafety()),
                effort_handling=tm.to_dict(tm.EffortClip()),
                cases={key: tm.to_dict(law) for key, law in cases().items()},
                case_params={key: tm.to_dict(case_params(key, params, width)) for key in cases()},
                assumptions='Virtual time, periodic held feedback, no additional sensor/command transport delay '
                            'or noise. ' + delay + 'Fitted j1-4, nominal wrists; '
                            'one holding/sliding threshold, no SEA dynamics. Supplied armature in plant and '
                            'all predictors. External FF width .03; internal MPC width recorded per case. '
                            'Existing PD gains unchanged; no external FF on friction-in-MPC cases.')


def export_and_measure(record, destination, case):
    """Keep observed results, add true motion columns/results and export executed truth."""
    evaluate(record)  # measured-state metrics remain independently available
    ticks, plans = record.ticks, record.plans
    truth = [diagnostic for diagnostic in record.diagnostics if diagnostic.kind == 'plant_state']
    if len(ticks['t']) == 0:
        return dict(case=case, status=record.metadata['status'], samples=0, replay=None)
    np.testing.assert_allclose([event.t for event in truth], ticks['t'], rtol=0, atol=1e-9)
    q = np.asarray([event.data['q'] for event in truth])
    qd = np.asarray([event.data['qd'] for event in truth])
    ticks['plant_q'], ticks['plant_qd'] = q, qd
    task, params = record.config['task'], record.config['params']
    ticks['plant_ee_xyz_m'] = np.array([tm.ee_position(position) for position in q])
    error = np.linalg.norm(ticks['plant_ee_xyz_m'] - task.path(ticks['t']), axis=1)
    quality = metrics(error)
    excess = limit_excess(dict(q=q, qd=qd, command=ticks['command']))
    results = record.metadata['results']
    results.update(plant_metrics_full_cycle=quality, plant_max_limit_excess=excess,
                   plant_limits_pass=max(excess.values()) <= 1e-5,
                   observation_age_max_s=float(np.max(ticks['t'] - ticks['state_t'])))
    record.save()
    # The standard export depicts the held feedback signal on command-tick times.
    observed = record.out / 'replay.npz'
    if observed.exists():
        observed.rename(record.out / 'held-feedback-replay.npz')
    if len(ticks['t']) < 2:
        return dict(case=case, status=record.metadata['status'], samples=1, replay=None)
    predictions = [dict(origin_time_s=float(t0), time_s=np.arange(params.N)*params.dt_s,
                        q=states[:, :7], qd_rad_s=states[:, 7:], torque_nm=u,
                        torque_time_s=np.arange(params.N-1)*params.dt_s)
                   for t0, states, u in zip(plans['t0'], plans['states'], plans['torques'])]
    sent = np.isfinite(ticks['command']).all(axis=1)
    raw = tm.limits()
    complete = record.metadata['status'] == 'done'
    label = LABELS[case]
    if case.startswith('mpc_friction'):
        label += f' [w={params.friction_law.width_rad_s:g}]'
    save_replay(record.out / 'replay.npz', time_s=ticks['t'], q=q, qd_rad_s=qd,
                joint_names=tm.RIGHT_ARM_JOINTS, time_basis='simulation', end_effector_frame=tm.GRASP_FRAME,
                label=label + ('' if complete else ' (stopped early)'),
                torque_nm=ticks['command'][sent] if sent.any() else None,
                torque_time_s=ticks['t'][sent] if sent.any() else None,
                plans=predictions, reference_xyz_m=task.path(ticks['t']), reference_time_s=ticks['t'],
                reference_frame=tm.ROOT_FRAME,
                joint_limits=dict(lower_position_rad=raw.lower, upper_position_rad=raw.upper,
                                  velocity_rad_s=raw.velocity, effort_nm=raw.effort,
                                  source='Canonical URDF SHA256 ' + hashlib.sha256(tm.FULL_URDF.read_bytes()).hexdigest()),
                provenance=dict(config=tm.to_dict(record.config), params=tm.to_dict(params),
                                source=record.config.get('dependency_source', record.provenance),
                                recorded_source=record.provenance, outcome=record.metadata,
                                raw_run=str(record.out.resolve()), motion='true simulated state, not held feedback',
                                sample_conversion='none'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(record.out / 'replay.npz', destination)
    return dict(case=case, label=label, status=record.metadata['status'], reason=record.metadata['reason'],
                end_s=float(ticks['t'][-1]), rmse_mm=quality['rmse_m']*1000,
                max_error_mm=quality['max_m']*1000, metric_scope='full cycle' if complete else 'recorded prefix',
                sampled_limits_pass=results['plant_limits_pass'], finite_states=bool(np.isfinite(q).all() and np.isfinite(qd).all()),
                samples=len(q), plans=len(plans['t0']), observation_age_max_s=results['observation_age_max_s'],
                output=str(record.out.resolve()), replay=str(destination))


def main(argv=None):
    handle_stop_signals()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--duration', type=float, default=24.)
    parser.add_argument('--mu', type=float, default=PARAMS.solver.mu)
    parser.add_argument('--qd-cost', type=float, default=float(np.asarray(PARAMS.cost.state.weight)[7]))
    parser.add_argument('--u-cost', type=float, default=float(np.asarray(PARAMS.cost.input.weight)))
    parser.add_argument('--rho-u', type=float, default=PARAMS.admm.rho_u)
    parser.add_argument('--dt', type=float, default=PARAMS.dt_s)
    parser.add_argument('--terminal-ee-cost', type=float, default=float(np.asarray(PARAMS.cost.frame_pos.weight)[-1, 0]))
    parser.add_argument('--command-period', type=float, default=TIMING.command_period)
    parser.add_argument('--update-period', type=float, default=TIMING.update_period)
    parser.add_argument('--solve-latency', type=parse_solve_latency, default=TIMING.solve_latency_s)
    parser.add_argument('--feedback-period', type=float, default=FEEDBACK_PERIOD_S)
    parser.add_argument('--mpc-friction-width', type=float, default=MPC_FRICTION_WIDTH_RAD_S)
    parser.add_argument('--cases', nargs='+', choices=tuple(cases()), help='run only the named cases')
    args = parser.parse_args(argv)
    params = replace(PARAMS, dt_s=args.dt,
                     solver=replace(PARAMS.solver, mu=args.mu),
                     cost=tm.default_cost(PARAMS.N, velocity=args.qd_cost, torque=args.u_cost,
                                          terminal_position=args.terminal_ee_cost),
                     admm=replace(PARAMS.admm, rho_u=args.rho_u))
    plant = plant_config(args.feedback_period)
    timing = tm.VirtualTimeSettings(args.command_period, args.update_period, args.solve_latency)
    if timing.solve_latency_s != 'measured':
        params = replace(params, max_sqp_time_s=None)  # match run_experiment's effective parameters
    config = configuration(params, plant, timing, args.mpc_friction_width)
    selected = args.cases if args.cases is not None else list(cases())
    config['selected_cases'] = selected
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0
    if args.output is None:
        parser.error('--output is required without --dry-run')
    out = args.output.resolve()
    if not out.is_relative_to(tm.ROOT / '_expr_data'):
        parser.error('--output must be inside this checkout\'s _expr_data')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'configuration.json').write_text(json.dumps(config, indent=2) + '\n')
    for source in (Path(__file__), Path(__file__).with_name('plan_feedback.py')):
        shutil.copyfile(source, out / source.name)
    ingest = tm.ROOT.parent / '_expr_data/ingest/friction-compensation' / out.name
    locks = []
    try:
        for path in (tm.ROOT.parent / 'GATO-upstream-contributions/.gpu.lock', Path('/tmp/a2rlab-timing.lock'),
                     tm.ROOT.parent / 'GATO/.git/experiment-runner.lock'):
            if path.parent.exists():
                lock = path.open('a')
                locks.append(lock)
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for name, law in cases().items():
            if name not in selected:
                continue
            try:
                record = run_experiment(plant, case_params(name, params, args.mpc_friction_width),
                                        Figure8(), law, timing, compensation=None, duration=args.duration,
                                        out=out / name, case=name, dependency_source=config['provenance'])
            except Exception:
                traceback.print_exc()
                rows.append(dict(case=name, status='error', output=str(out / name)))
            else:
                rows.append(export_and_measure(record, ingest / (name + '.npz'), name))
            (out / 'rows.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(json.dumps(rows[-1]), flush=True)
            if tm.STOP.is_set():
                break
    finally:
        for lock in locks:
            lock.close()
    return 0 if all(row['status'] in ('done', 'timeout') and row.get('sampled_limits_pass') for row in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
