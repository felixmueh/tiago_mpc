"""Native GATO menu construction and explicit grasp-frame targets."""
import pickle
from unittest.mock import Mock

from gato.cost_menu import CostMenu, LinearTerm, Term
import numpy as np
import pinocchio as pin
import pytest

from tiago_mpc.costs import default_cost, with_reference


REFERENCE = np.array([[.3, -.2, .1, np.nan, np.nan, np.nan],
                      [.3, -.2, .2, 0., 0., .4],
                      [.3, -.2, .3, .1, 0., 0.]])


def test_default_cost_is_a_native_menu_with_zero_and_identity_targets():
    menu = default_cost(3, orientation=.5, terminal_orientation=2.)
    assert type(menu) is CostMenu
    assert set(menu.terms()) == {"frame_pos", "state", "input", "orientation"}
    np.testing.assert_array_equal(menu.frame_pos.weight, [[4.] * 3, [4.] * 3, [8.] * 3])
    np.testing.assert_array_equal(menu.frame_pos.target, np.zeros(3))
    np.testing.assert_array_equal(menu.state.weight, [0.] * 7 + [.05] * 7)
    assert menu.state.target is None and menu.input.weight == 1e-4 and menu.input.target is None
    np.testing.assert_array_equal(menu.orientation.weight, [[.5] * 3, [.5] * 3, [2.] * 3])
    np.testing.assert_array_equal(menu.orientation.target, np.eye(3))
    assert default_cost() == default_cost()


def test_tracking_weights_and_rotations_follow_each_reference_knot():
    menu = with_reference(default_cost(3, orientation=.5, terminal_orientation=2.), REFERENCE)
    np.testing.assert_array_equal(menu.frame_pos.target, REFERENCE[:, :3])
    np.testing.assert_array_equal(menu.orientation.weight, [[0.] * 3, [.5] * 3, [2.] * 3])
    np.testing.assert_array_equal(menu.orientation.target[0], np.eye(3))
    np.testing.assert_allclose(menu.orientation.target[1], pin.exp3(REFERENCE[1, 3:]))
    np.testing.assert_allclose(menu.orientation.target[2], pin.exp3(REFERENCE[2, 3:]))


def test_fractional_terminal_weights_survive_integer_running_weights():
    template = default_cost(3, position=1, terminal_position=.25, orientation=0, terminal_orientation=.5)
    menu = with_reference(template, REFERENCE)
    np.testing.assert_array_equal(menu.frame_pos.weight, [[1.] * 3, [1.] * 3, [.25] * 3])
    np.testing.assert_array_equal(menu.orientation.weight, [[0.] * 3, [0.] * 3, [.5] * 3])
    np.testing.assert_allclose(menu.orientation.target, pin.exp3(REFERENCE[-1, 3:]))


def test_constant_orientation_converts_once_and_broadcasts_its_matrix(monkeypatch):
    reference = REFERENCE.copy()
    reference[1:, 3:] = [.1, .2, .3]
    exp3 = Mock(wraps=pin.exp3)
    monkeypatch.setattr(pin, "exp3", exp3)
    menu = with_reference(default_cost(3, orientation=1., terminal_orientation=2.), reference)
    exp3.assert_called_once()
    assert menu.orientation.target.shape == (3, 3)
    np.testing.assert_array_equal(menu.orientation.weight, [[0.] * 3, [1.] * 3, [2.] * 3])


def test_absent_orientation_or_untargeted_knots_need_no_rotation_conversion(monkeypatch):
    exp3 = Mock(side_effect=AssertionError("zero orientation must not be converted"))
    monkeypatch.setattr(pin, "exp3", exp3)
    assert with_reference(default_cost(3), REFERENCE).orientation is None
    untargeted = REFERENCE.copy()
    untargeted[:, 3:] = np.nan
    menu = with_reference(default_cost(3, orientation=1., terminal_orientation=2.), untargeted)
    np.testing.assert_array_equal(menu.orientation.weight, np.zeros((3, 3)))
    np.testing.assert_array_equal(menu.orientation.target, np.eye(3))
    untargeted[0, 3:] = [.1, .2, .3]
    menu = with_reference(default_cost(3, terminal_orientation=1.), untargeted)
    np.testing.assert_array_equal(menu.orientation.weight, np.zeros((3, 3)))
    exp3.assert_not_called()


@pytest.mark.parametrize("enabled, term", [(dict(position=1.), "frame_pos"),
                                         (dict(terminal_position=1.), "frame_pos"),
                                         (dict(velocity=1.), "state"),
                                         (dict(torque=1.), "input"),
                                         (dict(orientation=1.), "orientation"),
                                         (dict(terminal_orientation=1.), "orientation")])
def test_zero_weight_terms_are_omitted(enabled, term):
    weights = dict(position=0., terminal_position=0., velocity=0., torque=0.) | enabled
    assert set(default_cost(3, **weights).terms()) == {term}


def test_an_all_zero_objective_keeps_one_zero_term_to_make_a_native_menu():
    menu = default_cost(3, position=0., terminal_position=0., velocity=0., torque=0.)
    assert set(menu.terms()) == {"input"}
    assert menu.input.weight == 0.


def test_reference_updates_preserve_template_and_own_their_new_targets():
    template = default_cost(3, orientation=1., terminal_orientation=2.)
    reference = REFERENCE.copy()
    before, original = pickle.dumps(template), reference.copy()
    menu = with_reference(template, reference)
    np.testing.assert_array_equal(reference, original)
    reference[:] = 0.
    np.testing.assert_array_equal(menu.frame_pos.target, original[:, :3])
    np.testing.assert_allclose(menu.orientation.target[1], pin.exp3(original[1, 3:]))
    restored = with_reference(template, reference)
    np.testing.assert_array_equal(restored.orientation.weight, [[1.] * 3, [1.] * 3, [2.] * 3])
    assert pickle.dumps(template) == before


def test_native_terms_and_activation_settings_are_preserved():
    template = CostMenu(frame_pos=Term(weight=[1., 2., 3.], activation="pseudo_huber", eps=.02),
                        orientation=Term(weight=.5, target=np.eye(3), activation="pseudo_huber", eps=.03),
                        state=Term(weight=1., target=np.arange(14)),
                        input=Term(weight=1., target="gravity"), q_ref=np.zeros((3, 7)),
                        linear_u=LinearTerm(weight=2., A=np.eye(7), b=np.arange(7)),
                        velocity=Term(weight=1., target=np.arange(6)),
                        frame_pos_2=Term(weight=2., target=np.ones(3)))
    before = pickle.dumps(template)
    menu = with_reference(template, REFERENCE)
    assert menu.frame_pos.activation == "pseudo_huber" and menu.frame_pos.eps == .02
    assert menu.orientation.activation == "pseudo_huber" and menu.orientation.eps == .03
    for name in ("state", "input", "linear_u", "velocity", "frame_pos_2", "q_ref"):
        assert getattr(menu, name) is getattr(template, name)
    assert pickle.dumps(template) == before


@pytest.mark.parametrize("weight", [.5, [1., 2., 3.], np.arange(9).reshape(3, 3),
                                   np.arange(18).reshape(2, 3, 3)])
def test_native_orientation_weight_shapes_support_nan_masks(weight):
    menu = with_reference(CostMenu(orientation=Term(weight=weight, target=np.eye(3))), REFERENCE)
    expected = np.broadcast_to(weight, (3, 3) if np.ndim(weight) < 3 else (2, 3, 3)).copy()
    expected[..., 0, :] = 0.
    np.testing.assert_array_equal(menu.orientation.weight, expected)


@pytest.mark.parametrize("reference", [np.zeros((2, 5)), np.zeros((1, 6)), np.zeros(6),
                                       np.full((3, 6), np.nan),
                                       np.c_[np.zeros((3, 3)), [[np.nan, 0., 0.]] * 3],
                                       np.c_[np.zeros((3, 3)), [[np.inf, 0., 0.]] * 3]])
def test_malformed_references_are_rejected(reference):
    with pytest.raises(ValueError):
        with_reference(default_cost(3, orientation=1.), reference)


@pytest.mark.parametrize("field", ["position", "terminal_position", "velocity", "torque",
                                   "orientation", "terminal_orientation"])
@pytest.mark.parametrize("weight", [-1., np.nan, np.inf])
def test_weights_are_finite_and_nonnegative(field, weight):
    with pytest.raises(ValueError, match="weights"):
        default_cost(**{field: weight})


@pytest.mark.parametrize("n", [1, 0, -1, 3.5])
def test_horizon_has_at_least_two_integer_knots(n):
    with pytest.raises(ValueError, match="n must"):
        default_cost(n)
