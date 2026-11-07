"""Bounded simulation tuning of execution PD and external friction feedforward.

Explicit tasks, laws and coefficients; MPC costs, ADMM and solver remain fixed.
Run --dry-run first. --laws and --tasks select a small batch, not a search service.
All raw runs and failures survive. This module never connects to a robot.
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
from experiments.figure8 import Figure8, metrics
from experiments.friction_compensation import (PARAMS, TIMING,
                                              PLANT_COULOMB_J1_4_NM, PLANT_VISCOUS_J1_4_NM_S_RAD,
                                              SampledSimConfig, case_params, plant_config)
from experiments.motion_metrics import high_frequency_rms, motion_metrics, planned_velocity
from experiments.replay import save_replay
import tiago_mpc as tm
from tiago_mpc.tiago import ARMATURE_KG_M2


@dataclass(frozen=True)
class HoldPosition:
    """Position step held for a fixed duration, including settling and residual error."""
    target_m: tuple[float, float, float]
    duration_s: float = 6.

    def at(self, t, n, dt):
        out = np.full((n, 6), np.nan, dtype=np.float32)  # NaN orientation: none
        out[:, :3] = self.target_m
        return out, None

    def done(self, state, t):
        return t >= self.duration_s - 1e-9


def tasks() -> dict[str, Figure8 | HoldPosition]:
    start = tm.ee_position(tm.Q_START)
    return {'figure8_24': Figure8(), 'reach_x': HoldPosition(tuple(start + (.10, 0., 0.))),
            'figure8_12': Figure8(period_s=12.),
            'reach_yz': HoldPosition(tuple(start + (.04, -.08, .06)))}


def feedback(bandwidth: float, damping: float = 1.) -> tm.PlanFeedback:
    return tm.PlanFeedback(*tm.inertia_gains(bandwidth, damping, armature=ARMATURE_KG_M2))


def estimate(scale: float = 1., source: str = 'thesis') -> tm.Friction:
    nominal = tm.urdf_friction()
    prior = (replace(nominal, coulomb_nm=PLANT_COULOMB_J1_4_NM + nominal.coulomb_nm[4:],
                     viscous_nm_s_per_rad=PLANT_VISCOUS_J1_4_NM_S_RAD + nominal.viscous_nm_s_per_rad[4:])
             if source == 'thesis' else nominal)
    return tm.Friction(tuple(scale*np.asarray(prior.coulomb_nm)),
                       tuple(scale*np.asarray(prior.viscous_nm_s_per_rad)))


# Locked from the two training tasks before any held-out task/stress runs.
# Simulation presets, not hardware-qualified gains. External width only; the
# 30 ms friction-in-MPC predictor keeps its separately qualified width .2.
TUNED_PD = tm.PlanFeedback()
TUNED_FEEDFORWARD = tm.FrictionFeedforward(estimate(), evaluation=tm.Tanh(.01))
TUNED_BOTH = tm.FrictionFeedforward(estimate(), TUNED_PD, evaluation=tm.Tanh(.01))


def laws() -> dict[str, tm.ControlLaw]:
    pd = {'pd10': feedback(10.), 'pd20': feedback(20.), 'pd30': feedback(30.),
          'pd20_z07': feedback(20., .7), 'pd20_z14': feedback(20., 1.4)}
    ff = {'ff_prior01': (estimate(), .01), 'ff_prior03': (estimate(), .03),
          'ff_prior10': (estimate(), .1), 'ff_half03': (estimate(scale=.5), .03),
          'ff_half10': (estimate(scale=.5), .1), 'ff_nominal10': (estimate(source='nominal'), .1)}
    old_pd = tm.PlanFeedback(*tm.inertia_gains(30.))
    result = {'default': tm.SampleKnot(), 'mpc_friction': tm.SampleKnot(), 'mpc_friction_pd': TUNED_PD,
              'old_pd': old_pd,
              'old_ff': tm.FrictionFeedforward(estimate(source='nominal')),
              'old_both': tm.FrictionFeedforward(estimate(source='nominal'), old_pd)}
    result.update(pd)
    result.update({key: tm.FrictionFeedforward(value, evaluation=tm.Tanh(width))
                   for key, (value, width) in ff.items()})
    result.update({p + '+' + f: tm.FrictionFeedforward(friction, law, evaluation=tm.Tanh(width))
                   for p, law in pd.items() for f, (friction, width) in ff.items()})
    return result


def scenario(name: str, feedback_period: float = .002) -> SampledSimConfig:
    plant = plant_config(feedback_period)
    if name == 'nominal':
        return plant
    if name == 'no_friction':
        return replace(plant, physical=replace(plant.physical, friction=None))
    # Deliberate mismatch around the measured-proximal / nominal-wrist plant.
    loss_scale, inertia_scale = {'low_loss': (.5, 1.25), 'high_loss': (1.5, .75)}[name]
    friction = plant.physical.friction
    friction = replace(friction, coulomb_nm=tuple(loss_scale*np.asarray(friction.coulomb_nm)),
                       viscous_nm_s_per_rad=tuple(loss_scale*np.asarray(friction.viscous_nm_s_per_rad)))
    return replace(plant, physical=replace(plant.physical, friction=friction,
                                          armature=tuple(inertia_scale*np.asarray(ARMATURE_KG_M2))))


def evaluate(record: tm.RunRecord) -> None:
    """Measure executed truth, preserve observations, and save the task results."""
    label = record.config['case'] + ' / ' + record.config['task_name'] + ' / ' + record.config['scenario']
    ticks, plans, params, task = record.ticks, record.plans, record.config['params'], record.config['task']
    truth = [e for e in record.diagnostics if e.kind == 'plant_state']
    if not len(ticks['t']):
        record.metadata['results']['tuning'] = dict(label=label, status=record.metadata['status'],
                                                    valid=False, output=str(record.out))
        record.save()
        return
    np.testing.assert_allclose([e.t for e in truth], ticks['t'], atol=1e-9, rtol=0)
    q, qd = np.array([e.data['q'] for e in truth]), np.array([e.data['qd'] for e in truth])
    xyz = np.array([tm.ee_position(position) for position in q])
    reference = np.array([task.at(t, 2, params.dt_s)[0][0, :3] for t in ticks['t']])
    error = np.linalg.norm(xyz-reference, axis=1)
    sent = np.isfinite(ticks['command']).all(axis=1)
    torque = ticks['command'][sent]
    raw = tm.limits()
    excess = limit_excess(dict(q=q, qd=qd, command=ticks['command']))
    dt = record.config['settings'].command_period
    late = ticks['t'] >= 1.
    tail = ticks['t'] >= max(0., ticks['t'][-1]-1.)
    quality = metrics(error)
    row = dict(label=label, status=record.metadata['status'], reason=record.metadata['reason'],
               valid=record.metadata['status'] == 'done' and max(excess.values()) <= 1e-5,
               end_s=float(ticks['t'][-1]), rmse_mm=1000*quality['rmse_m'],
               after_1s_rmse_mm=1000*metrics(error[late])['rmse_m'] if late.any() else None,
               tail_rmse_mm=1000*metrics(error[tail])['rmse_m'], max_error_mm=1000*quality['max_m'],
               final_error_mm=1000*float(error[-1]), final_speed_m_s=float(np.linalg.norm(tm.ee_velocity(q[-1], qd[-1]))),
               max_limit_excess=excess, samples=len(q), plans=len(plans['t0']),
               torque_fraction_max=float(np.max(np.abs(torque)/raw.effort)) if len(torque) else None,
               torque_hf_rms_nm=high_frequency_rms(torque, dt).tolist(),
               velocity_hf_rms_rad_s=high_frequency_rms(qd, dt).tolist(),
               max_velocity_rad_s=np.max(np.abs(qd), axis=0).tolist(),
               observation_age_max_s=float(np.max(ticks['t']-ticks['state_t'])), output=str(record.out.resolve()))
    intended = planned_velocity(ticks, plans, params.dt_s)
    row['motion_full'] = motion_metrics(ticks['t'], qd, intended, ticks['command'])
    row['motion_after_1s'] = motion_metrics(ticks['t'], qd, intended, ticks['command'], start_s=1.)
    row['after_1s_error_ripple_above_1hz_rms_mm'] = (
        1000 * float(np.linalg.norm(high_frequency_rms(xyz[late] - reference[late], dt, 1.)))
        if late.sum() >= 3 else None)
    ticks.update(plant_q=q, plant_qd=qd, plant_ee_xyz_m=xyz, reference_xyz_m=reference)
    ticks['planned_qd'] = intended
    record.metadata['results'].update(tuning=row)
    record.save()


def export(record: tm.RunRecord, destination: Path) -> None:
    """Export the separately measured true motion with all plans and provenance."""
    ticks, plans, params = record.ticks, record.plans, record.config['params']
    q, qd, reference = ticks['plant_q'], ticks['plant_qd'], ticks['reference_xyz_m']
    sent = np.isfinite(ticks['command']).all(axis=1)
    torque, raw = ticks['command'][sent], tm.limits()
    label = record.metadata['results']['tuning']['label']
    if len(q) > 1:
        existing = record.out/'replay.npz'
        if existing.exists():
            existing.rename(record.out/'observed-replay.npz')
        predictions = [dict(origin_time_s=float(t0), time_s=np.arange(params.N)*params.dt_s,
                            q=state[:, :7], qd_rad_s=state[:, 7:], torque_nm=u,
                            torque_time_s=np.arange(params.N-1)*params.dt_s)
                       for t0, state, u in zip(plans['t0'], plans['states'], plans['torques'])]
        save_replay(existing, time_s=ticks['t'], q=q, qd_rad_s=qd, joint_names=tm.RIGHT_ARM_JOINTS,
                    time_basis='simulation', end_effector_frame=tm.GRASP_FRAME, label=label,
                    torque_nm=torque if sent.any() else None, torque_time_s=ticks['t'][sent] if sent.any() else None,
                    plans=predictions, reference_xyz_m=reference, reference_time_s=ticks['t'],
                    reference_frame=tm.ROOT_FRAME,
                    joint_limits=dict(lower_position_rad=raw.lower, upper_position_rad=raw.upper,
                                      velocity_rad_s=raw.velocity, effort_nm=raw.effort, source='canonical URDF'),
                    provenance=dict(config=tm.to_dict(record.config), params=tm.to_dict(params),
                                    source=record.config['dependency_source'], recorded_source=record.provenance,
                                    outcome=record.metadata, raw_run=str(record.out.resolve()),
                                    motion='true simulated state; virtual time; no noise or transport delay'))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(existing, destination)


def main() -> int:
    handle_stop_signals()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--laws', nargs='+', choices=tuple(laws()), required=True)
    parser.add_argument('--tasks', nargs='+', choices=tuple(tasks()), required=True)
    parser.add_argument('--scenario', choices=('nominal', 'no_friction', 'low_loss', 'high_loss'), default='nominal')
    parser.add_argument('--dt', type=float, default=.03)
    parser.add_argument('--feedback-period', type=float, default=.002)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--solve-latency', type=parse_solve_latency, default='measured')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(tm.ROOT/'_expr_data'):
        parser.error('output must be inside this checkout\'s _expr_data')
    params = replace(PARAMS, dt_s=args.dt, max_sqp_time_s=None)
    plant = scenario(args.scenario, args.feedback_period)
    timing = replace(TIMING, solve_latency_s=args.solve_latency)
    config = dict(provenance=tm.provenance(), params=tm.to_dict(params), plant=tm.to_dict(plant),
                  settings=tm.to_dict(timing), loop="run_virtual_time", compensation=None,
                  safety=tm.to_dict(tm.ExecutionSafety()), effort_handling=tm.to_dict(tm.EffortClip()),
                  laws={k: tm.to_dict(laws()[k]) for k in args.laws},
                  case_params={k: tm.to_dict(case_params(k, params)) for k in args.laws},
                  tasks={k: tm.to_dict(tasks()[k]) for k in args.tasks}, duration_cap_s=args.duration,
                  urdf_sha256=hashlib.sha256(tm.FULL_URDF.read_bytes()).hexdigest(),
                  purpose='Execution PD / friction feedforward tuning; solver unchanged; simulation only')
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=False)
    (out/'configuration.json').write_text(json.dumps(config, indent=2)+'\n')
    for path in (__file__, Path(__file__).with_name('motion_metrics.py'), tm.ROOT/'tiago_mpc/laws.py',
                 Path(__file__).with_name('friction_compensation.py')):
        shutil.copyfile(path, out/Path(path).name)
    locks = []
    try:
        for path in (tm.ROOT.parent/'GATO-upstream-contributions/.gpu.lock', Path('/tmp/a2rlab-timing.lock'),
                     tm.ROOT.parent/'GATO/.git/experiment-runner.lock'):
            if path.parent.exists():
                lock = path.open('a')
                locks.append(lock)
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for task_name in args.tasks:
            for name in args.laws:
                key = task_name + '/' + name
                try:
                    record = run_experiment(plant, case_params(name, params), tasks()[task_name], laws()[name], timing,
                                            duration=args.duration, out=out/key, case=name, task_name=task_name,
                                            scenario=args.scenario, dependency_source=config['provenance'])
                    evaluate(record)
                    row = dict(record.metadata['results']['tuning'])
                    if len(record.ticks['t']) > 1:
                        study = out.relative_to(tm.ROOT/'_expr_data')
                        destination = tm.ROOT.parent/'_expr_data/ingest/controller-tuning'/study/(key+'.npz')
                        export(record, destination)
                        row['replay'] = str(destination)
                except Exception:
                    traceback.print_exc()
                    row = dict(status='error', valid=False, output=str(out/key))
                row.update(law=name, task=task_name, scenario=args.scenario, dt_s=args.dt)
                rows.append(row)
                (out/'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
                print(json.dumps(row), flush=True)
                if tm.STOP.is_set():
                    return 1
    finally:
        for lock in locks:
            lock.close()
    return 0 if all(r['status'] in ('done', 'timeout') and max(r.get('max_limit_excess', {'none': 1.}).values()) <= 1e-5
                    for r in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
