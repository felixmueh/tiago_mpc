# Adapted from GATO ros-tiago c4cf5cc tests; no ROS services are contacted.
"""Exercise real mode switching against a controller-manager service double."""
from types import SimpleNamespace as NS

import pytest

from tiago_mpc.ros2 import TiagoRightArmClient
from tiago_mpc.tiago import RIGHT_ARM_JOINTS

POSITION = "arm_right_controller"
GRAVITY = "arm_right_gravity_compensation_controller"
EFFORT = "gato_arm_effort_forward_runtime"


def make_arm(active=(POSITION,), *, bad_claims=False, switch_ok=True, apply=True):
    arm = TiagoRightArmClient.__new__(TiagoRightArmClient)
    arm.joint_names = RIGHT_ARM_JOINTS
    arm.controller_manager = "/controller_manager"
    arm.effort_controller = EFFORT
    arm.trajectory_topic = "/arm_right_controller/joint_trajectory"
    arm.effort_command_topic = f"/{EFFORT}/commands"
    arm.ros = {name: NS(Request=NS) for name in ("ListControllers", "SwitchController", "GetParameters")}
    arm.ros["ParameterType"] = NS(PARAMETER_STRING=4, PARAMETER_STRING_ARRAY=9)
    arm.ros["Duration"] = NS
    controllers = {}
    calls = []
    for name in (POSITION, GRAVITY, EFFORT):
        interface = "position" if name == POSITION else "effort"
        controllers[name] = NS(name=name, type="forward_command_controller/ForwardCommandController", state="active" if name in active else "inactive",
            claimed_interfaces=[f"{j}/{interface}" for j in RIGHT_ARM_JOINTS]
            if name in active else [], required_command_interfaces=[
                f"{j}/{interface}" for j in RIGHT_ARM_JOINTS])
    if bad_claims:
        controllers[POSITION].claimed_interfaces = []

    parameters = {"joints": NS(type=9, string_array_value=list(RIGHT_ARM_JOINTS)),
                  "interface_name": NS(type=4, string_value="effort")}
    arm.test_parameters = parameters

    def service(name, kind, request, timeout):
        if name.endswith("get_parameters"):
            assert name == f"/{EFFORT}/get_parameters"
            return NS(values=[parameters[key] for key in request.names])
        if name.endswith("list_controllers"):
            return NS(controller=list(controllers.values()))
        calls.append(request)
        if switch_ok and apply:
            for n in request.deactivate_controllers:
                controllers[n].state = "inactive"
                controllers[n].claimed_interfaces = []
            for n in request.activate_controllers:
                controllers[n].state = "active"
                controllers[n].claimed_interfaces = controllers[n].required_command_interfaces
        return NS(ok=switch_ok, message="switch rejected")

    arm._service_call = service
    arm._set_remote_parameters = lambda *a, **k: None
    # Inactive controllers also have subscribers: this must not decide mode.
    arm._topic_has_subscription = lambda _: True
    arm._wait_for_topic_subscription = lambda *a: None
    return arm, controllers, calls


def test_round_trip_is_exclusive_and_idempotent():
    arm, controllers, calls = make_arm()
    arm.switch_to_default_control()
    assert calls == []
    arm.switch_to_effort_control()
    arm.switch_to_effort_control()
    arm.switch_to_default_control()
    arm.switch_to_default_control()
    assert len(calls) == 2
    assert calls[0].activate_controllers == [EFFORT]
    assert calls[0].deactivate_controllers == [POSITION]
    assert calls[1].activate_controllers == [POSITION]
    assert calls[1].deactivate_controllers == [EFFORT]
    assert all(c.strictness == 2 for c in calls)
    assert controllers[GRAVITY].state == "inactive"


@pytest.mark.parametrize("method", ["switch_to_default_control", "switch_to_effort_control"])
def test_mixed_mode_is_rejected_without_mutation(method):
    arm, _, calls = make_arm((POSITION, GRAVITY))
    with pytest.raises(RuntimeError, match="multiple.*active"):
        getattr(arm, method)()
    assert calls == []


def test_wrong_claims_cannot_pass_as_position_mode():
    arm, _, calls = make_arm(bad_claims=True)
    with pytest.raises(RuntimeError, match="claimed interfaces"):
        arm.switch_to_default_control()
    assert calls == []


@pytest.mark.parametrize("switch_ok,apply", [(False, True), (True, False)])
def test_switch_rejection_or_false_success_is_not_accepted(switch_ok, apply):
    arm, _, _ = make_arm(switch_ok=switch_ok, apply=apply)
    with pytest.raises(RuntimeError):
        arm.switch_to_effort_control()


def test_unknown_right_arm_owner_is_not_silently_deactivated():
    arm, controllers, calls = make_arm(())
    controllers["other"] = NS(name="other", state="active",
        claimed_interfaces=[f"{RIGHT_ARM_JOINTS[0]}/effort"])
    with pytest.raises(RuntimeError, match="unexpected.*other"):
        arm.switch_to_default_control()
    assert calls == []


def test_gravity_only_handover_uses_one_strict_request():
    arm, _, calls = make_arm((GRAVITY,))
    arm.switch_to_default_control()
    assert len(calls) == 1
    assert calls[0].activate_controllers == [POSITION]
    assert calls[0].deactivate_controllers == [GRAVITY]
    assert calls[0].strictness == 2


@pytest.mark.parametrize("bad_state", ["missing", "unconfigured", "finalized"])
def test_destination_must_be_loaded_and_configured(bad_state):
    arm, controllers, calls = make_arm((EFFORT,))
    if bad_state == "missing":
        del controllers[POSITION]
    else:
        controllers[POSITION].state = bad_state
    with pytest.raises(RuntimeError):
        arm.switch_to_default_control()
    assert calls == []


def test_inventory_with_duplicate_names_is_rejected():
    arm, controllers, calls = make_arm()
    arm._service_call = lambda *a: NS(controller=[controllers[POSITION]] * 2)
    with pytest.raises(RuntimeError, match="duplicate"):
        arm.switch_to_default_control()
    assert calls == []


def test_position_handback_holds_measured_state_before_reactivation():
    arm, _, _ = make_arm((EFFORT,))
    operations = []
    original_service = arm._service_call

    def service(name, kind, request, timeout):
        if name.endswith('switch_controller'):
            operations.append(('switch', list(request.activate_controllers)))
        return original_service(name, kind, request, timeout)
    arm._service_call = service
    arm._set_remote_parameters = lambda node, values, timeout: operations.append(
        ('parameters', node, values))
    arm.switch_to_default_control()
    assert operations == [
        ('parameters', '/arm_right_controller',
         {'set_last_command_interface_value_as_state_on_activation': False}),
        ('switch', [POSITION]),
    ]


@pytest.mark.parametrize("controller_state", ["inactive", "active"])
@pytest.mark.parametrize("fault", ["order", "missing_joint", "interface", "type", "unset"])
def test_existing_effort_controller_contract_is_checked_before_reuse(controller_state, fault):
    arm, controllers, calls = make_arm((EFFORT,) if controller_state == "active" else (POSITION,))
    parameters = arm.test_parameters
    if fault == "order":
        parameters["joints"].string_array_value.reverse()
    elif fault == "missing_joint":
        parameters["joints"].string_array_value.pop()
    elif fault == "interface":
        parameters["interface_name"].string_value = "position"
    elif fault == "type":
        controllers[EFFORT].type = "joint_trajectory_controller/JointTrajectoryController"
    elif fault == "unset":
        parameters["joints"].type = 0
    mutations = []
    arm._set_remote_parameters = lambda *args: mutations.append(args)
    with pytest.raises(RuntimeError, match="effort controller"):
        arm.configure_runtime_effort_controller()
    assert mutations == [] and calls == []


def test_reversed_effort_joint_order_cannot_be_activated():
    arm, controllers, calls = make_arm()
    arm.test_parameters["joints"].string_array_value.reverse()
    with pytest.raises(RuntimeError, match="joint order"):
        arm.switch_to_effort_control()
    assert calls == []


def test_correct_existing_effort_controller_is_reused_without_mutation():
    arm, _, calls = make_arm()
    arm._set_remote_parameters = lambda *args: pytest.fail("configured controller mutated")
    arm.configure_runtime_effort_controller()
    assert calls == []
