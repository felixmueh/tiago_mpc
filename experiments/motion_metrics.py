"""Descriptive tracking stalls, velocity ripple and command chatter from saved ticks.

Stalls mean sustained near-rest while the active plan requests motion, not proof
that friction caused it. Use true simulated velocity, not repeated observations.
Thresholds are explicit analysis choices, not identified robot noise thresholds.
"""
import numpy as np


def planned_velocity(ticks: dict, plans: dict, dt_s: float) -> np.ndarray:
    """Sample the plan actually active at each tick, keeping its original origin.

    Match the constant-acceleration interpolation used by the package laws. A
    pending/newer request is not active; ticks.plan is the authoritative index.
    Before activation and beyond the last torque interval, return NaN.
    """
    result = np.full((len(ticks['t']), 7), np.nan)
    indices = np.asarray(ticks['plan'], dtype=int)
    valid = (indices >= 0) & (indices < len(plans['t0']))
    rows = np.flatnonzero(valid)
    if not len(rows):
        return result
    indices = indices[rows]
    age = ticks['t'][rows] - plans['t0'][indices]
    knots = np.floor(age / dt_s + 1e-9).astype(int)
    inside = (age >= 0) & (knots >= 0) & (knots < plans['states'].shape[1] - 1)
    rows, indices, knots, age = rows[inside], indices[inside], knots[inside], age[inside]
    phase = np.clip((age - knots * dt_s) / dt_s, 0., 1.)[:, None]
    v0 = plans['states'][indices, knots, 7:]
    v1 = plans['states'][indices, knots + 1, 7:]
    result[rows] = v0 + phase * (v1 - v0)
    return result


def high_frequency_rms(values: np.ndarray, dt: float, cutoff_hz: float = 10.) -> np.ndarray:
    """RMS above cutoff on uniformly sampled data, after endpoint detrending."""
    if len(values) < 3:
        return np.zeros(values.shape[-1])
    values = values - np.linspace(values[0], values[-1], len(values))
    spectrum = np.fft.rfft(values, axis=0)
    spectrum[np.fft.rfftfreq(len(values), dt) <= cutoff_hz] = 0
    return np.sqrt(np.mean(np.fft.irfft(spectrum, n=len(values), axis=0)**2, axis=0))


def ripple_rms(t: np.ndarray, values: np.ndarray, valid: np.ndarray, cutoff_hz: float) -> list:
    """Combine contiguous uniform blocks; never concatenate missing command gaps."""
    rows = np.flatnonzero(valid & np.isfinite(values).all(axis=1))
    blocks = np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)
    energy, duration = np.zeros(values.shape[1]), 0.
    for block in blocks:
        if len(block) < 3:
            continue
        intervals = np.diff(t[block])
        dt = float(np.median(intervals))
        if dt <= 0 or not np.allclose(intervals, dt, rtol=1e-6, atol=1e-10):
            raise ValueError('spectral ripple requires uniform sampling within each recorded block')
        if not 0 <= cutoff_hz < .5 / dt:
            raise ValueError('ripple cutoff must be below the sampling Nyquist frequency')
        span = float(t[block[-1]] - t[block[0]])
        energy += span * high_frequency_rms(values[block], dt, cutoff_hz)**2
        duration += span
    return [None] * values.shape[1] if duration == 0 else np.sqrt(energy / duration).tolist()


def motion_metrics(
    t: np.ndarray,
    velocity: np.ndarray,
    intended_velocity: np.ndarray,
    command: np.ndarray,
    *,
    start_s: float = 0.,
    stopped_velocity_rad_s: float = .005,
    requested_velocity_rad_s: float = .02,
    minimum_stall_s: float = .05,
    ripple_cutoff_hz: float = 10.,
) -> dict:
    """Time-weighted per-joint and pooled stall metrics; no causal friction claim.

    Left tick values describe [t[i], t[i+1]); the final endpoint has no duration.
    A short zero crossing is not a stall. Pooling fractions uses eligible joint
    seconds; an unrequested joint contributes neither numerator nor denominator.
    Commands are measured separately: changing torque is not motion roughness.
    """
    t = np.asarray(t, dtype=float)
    velocity, intended_velocity, command = map(np.asarray, (velocity, intended_velocity, command))
    if (t.ndim != 1 or any(x.shape != (len(t), 7) for x in (velocity, intended_velocity, command))
            or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)):
        raise ValueError('metrics need increasing finite times and three (samples, 7) arrays')
    if not (0 <= stopped_velocity_rad_s < requested_velocity_rad_s and minimum_stall_s > 0):
        raise ValueError('stall thresholds require 0 <= stopped < requested and positive minimum duration')
    sent = np.isfinite(command).all(axis=1)
    weights = np.maximum(0., t[1:] - np.maximum(t[:-1], start_s))
    eligible = (np.isfinite(velocity[:-1]) & np.isfinite(intended_velocity[:-1])
                & (np.abs(intended_velocity[:-1]) >= requested_velocity_rad_s)
                & sent[:-1, None] & (weights[:, None] > 0))
    candidate = eligible & (np.abs(velocity[:-1]) <= stopped_velocity_rad_s)
    stalled_s, eligible_s, durations, per_joint = [], [], [], []
    for joint in range(7):
        edges = np.diff(np.r_[False, candidate[:, joint], False].astype(int))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        episodes = []
        for a, b in zip(starts, ends):
            duration = float(weights[a:b].sum())
            if duration < minimum_stall_s - 1e-9:
                continue
            left_censored = (a == 0 or t[a] <= start_s or not sent[a - 1]
                             or not np.isfinite(intended_velocity[a - 1, joint])
                             or not np.isfinite(velocity[a - 1, joint]))
            right_censored = (b == len(weights) or not sent[b]
                              or not np.isfinite(intended_velocity[b, joint])
                              or not np.isfinite(velocity[b, joint]))
            episodes.append(dict(start_s=float(max(t[a], start_s)), end_s=float(t[b]), duration_s=duration,
                                 left_censored=bool(left_censored), right_censored=bool(right_censored)))
        bouts = [event['duration_s'] for event in episodes]
        eligible_time = float(weights[eligible[:, joint]].sum())
        stopped_time = sum(bouts)
        eligible_s.append(eligible_time)
        stalled_s.append(stopped_time)
        durations.extend(bouts)
        per_joint.append(dict(eligible_s=eligible_time, stalled_s=stopped_time,
                              stalled_fraction=None if eligible_time == 0 else stopped_time / eligible_time,
                              count=len(bouts), median_s=float(np.median(bouts)) if bouts else None,
                              p95_s=float(np.percentile(bouts, 95)) if bouts else None,
                              max_s=max(bouts, default=None), episodes=episodes))
    total_eligible = sum(eligible_s)
    exact_zero_s = float((weights[:, None] * (eligible & (velocity[:-1] == 0))).sum())
    pairs = sent[:-1] & sent[1:] & (t[:-1] >= start_s)
    command_span = float(np.diff(t)[pairs].sum())
    variation = np.abs(np.diff(command, axis=0)[pairs]).sum(axis=0)
    late = (t >= start_s) & sent
    return dict(
        thresholds=dict(start_s=start_s, stopped_velocity_rad_s=stopped_velocity_rad_s,
                        requested_velocity_rad_s=requested_velocity_rad_s, minimum_stall_s=minimum_stall_s,
                        ripple_cutoff_hz=ripple_cutoff_hz),
        stall=dict(per_joint=per_joint, eligible_joint_s=total_eligible, stalled_joint_s=sum(stalled_s),
                   fraction=None if total_eligible == 0 else sum(stalled_s) / total_eligible,
                   count=len(durations), median_s=float(np.median(durations)) if durations else None,
                   p95_s=float(np.percentile(durations, 95)) if durations else None,
                   max_s=max(durations, default=None),
                   censored_count=sum(e['left_censored'] or e['right_censored']
                                      for joint in per_joint for e in joint['episodes'])),
        exact_zero_eligible_joint_s=exact_zero_s,
        exact_zero_eligible_fraction=None if total_eligible == 0 else exact_zero_s / total_eligible,
        velocity_ripple_rms_rad_s=ripple_rms(t, velocity, late, ripple_cutoff_hz),
        torque_ripple_rms_nm=ripple_rms(t, command, late, ripple_cutoff_hz),
        torque_total_variation_nm=variation.tolist(), command_pair_duration_s=command_span,
        torque_variation_nm_per_s=([None] * 7 if command_span == 0 else (variation / command_span).tolist()),
        interpretation='Plan-conditioned sustained near-rest; not proof of friction. Ripple above declared '
                       'cutoff; torque metrics describe command chatter, not arm motion.',
    )
