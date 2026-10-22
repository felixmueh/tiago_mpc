"""CPU regression checks for default promotion and frozen profile resolution."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiago_mpc.configuration import DEFAULT_CONFIG, load_configuration

LEGACY_SOLVER = dict(max_sqp_iters=5, max_pcg_iters=120, pcg_tol=.001,
                     solve_ratio=1., mu=10., rho=.01, adapt_rho=True, linsys=None,
                     q_cost=4., qd_cost=.05, u_cost=.0001, N_cost=80.,
                     q_lim_cost=.02125, vel_lim_cost=1.275, ctrl_lim_cost=.02,
                     exact_hessian=False)
LEGACY_ADMM = dict(rho=.01, iters=10, eps_rel=.01, rho_q=.1, rho_qd=.03, rho_u=.00001)


class ReachingConfigurationTests(unittest.TestCase):
    def test_selected_defaults_keep_separate_historical_resolution(self):
        path, settings, effective = load_configuration()
        self.assertEqual(path, DEFAULT_CONFIG)
        self.assertEqual(set(settings['solver_params']), set(LEGACY_SOLVER))
        self.assertEqual(effective['solver_params'], {
            **LEGACY_SOLVER, 'N_cost': 8., 'q_lim_cost': 0.,
            'vel_lim_cost': 0., 'ctrl_lim_cost': 0.})
        self.assertEqual((effective['N'], effective['dt_s']), (32, .01))
        self.assertEqual(effective['controller'], dict(
            warm_start='shift', reset_rho_each_step=True, admm_linsys='pcg',
            limits_kw=dict(rho=.01, iters=100, eps_rel=.001,
                           rho_q=1., rho_qd=.3, rho_u=.0005)))
        self.assertEqual(effective['limit_bounds'], dict(
            joint_margin_rad=.1, velocity_fraction=.8, effort_fraction=.85))
        self.assertEqual(effective['save_plan_steps'], 5)

    def test_explicit_empty_config_reproduces_original_example(self):
        data = ROOT / '_expr_data/config-tests'
        data.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=data) as tmp:
            path = Path(tmp) / 'original.json'
            path.write_text('{}\n')
            _, _, effective = load_configuration(path)
        self.assertEqual(effective['solver_params'], LEGACY_SOLVER)
        self.assertEqual(effective['dt_s'], .03)
        self.assertEqual(effective['controller']['limits_kw'], LEGACY_ADMM)
        self.assertEqual(effective['limit_bounds'], dict(
            joint_margin_rad=.1, velocity_fraction=None, effort_fraction=None,
            compiled_velocity_offset_rad_s=.1, compiled_effort_offset_nm=.1))

    def test_random100_dry_run_reports_selected_effective_defaults(self):
        raw = subprocess.check_output([
            sys.executable, str(ROOT / 'experiments/random100/run.py'), '--dry-run'],
            cwd=ROOT, text=True)
        config = json.loads(raw)
        self.assertEqual(config['configuration_file'], str(DEFAULT_CONFIG))
        self.assertEqual(config['effective_configuration'], load_configuration()[2])
        self.assertEqual(len(config['task_ids']), 100)
        self.assertEqual(config['end_effector_frame'], 'arm_right_tool_joint')

    def test_random100_copies_default_and_passes_snapshot_to_child(self):
        spec = importlib.util.spec_from_file_location('random100_run', ROOT / 'experiments/random100/run.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        data = ROOT / '_expr_data/config-tests'
        data.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=data) as tmp:
            out = Path(tmp) / 'cohort'
            def fake_episode(cmd, **kwargs):
                self.assertEqual(cmd[cmd.index('--ee-frame')+1], 'arm_right_tool_joint')
                copied = Path(cmd[cmd.index('--config')+1])
                self.assertEqual(copied, out.resolve() / 'config.json')
                self.assertEqual(copied.read_bytes(), DEFAULT_CONFIG.read_bytes())
                case = Path(cmd[cmd.index('--output')+1])
                case.mkdir()
                (case / 'summary.json').write_text(json.dumps(dict(
                    status='arrived', execution_stage='episode', reached=True, limits_pass=True)))
                return subprocess.CompletedProcess(cmd, 0)
            with patch.object(sys, 'argv', ['run.py', '--pairs', '8990', '--output', str(out)]), \
                    patch.object(runner.subprocess, 'check_output', return_value='test-source\n'), \
                    patch.object(runner.fcntl, 'flock'), \
                    patch.object(runner.subprocess, 'run', side_effect=fake_episode):
                self.assertEqual(runner.main(), 0)
            recorded = json.loads((out / 'configuration.json').read_text())
            self.assertEqual(recorded['effective_configuration'], load_configuration()[2])


if __name__ == '__main__':
    unittest.main()
