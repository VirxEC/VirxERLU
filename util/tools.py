from typing import Optional, Tuple, Union

import virx_erlu_rlib as rlru

from util.agent import VirxERLU
from util.routines import AerialShot, DoubleJumpShot, GroundShot, JumpShot
from util.utils import Vector

SHOT_SWITCH = (
    GroundShot,
    JumpShot,
    DoubleJumpShot,
    AerialShot,
)


def find_ground_shot(agent, target, cap_=6):
    return find_shot(agent, target, cap_, can_aerial=False, can_double_jump=False, can_jump=False)


def find_any_ground_shot(agent, cap_=6):
    return find_any_shot(agent, cap_, can_aerial=False, can_double_jump=False, can_jump=False)


def find_jump_shot(agent, target, cap_=6):
    return find_shot(agent, target, cap_, can_aerial=False, can_double_jump=False, can_ground=False)


def find_any_jump_shot(agent, cap_=6):
    return find_any_shot(agent, cap_, can_aerial=False, can_double_jump=False, can_ground=False)


def find_double_jump(agent, target, cap_=6):
    return find_shot(agent, target, cap_, can_aerial=False, can_jump=False, can_ground=False)


def find_any_double_jump(agent, cap_=6):
    return find_any_shot(agent, cap_, can_aerial=False, can_jump=False, can_ground=False)


def find_aerial(agent, target, cap_=6):
    return find_shot(agent, target, cap_, can_double_jump=False, can_jump=False, can_ground=False)


def find_any_aerial(agent, cap_=6):
    return find_any_shot(agent, cap_, can_double_jump=False, can_jump=False, can_ground=False)


def find_shot(agent: VirxERLU, target: Tuple[Vector, Vector], cap_: int=6, can_aerial: bool=True, can_double_jump: bool=True, can_jump: bool=True, can_ground: bool=True) -> Optional[Union[GroundShot, JumpShot, DoubleJumpShot, AerialShot]]:
    if not can_aerial and not can_double_jump and not can_jump and not can_ground:
        agent.print("WARNING: All shots were disabled when find_shot was ran")
        return

    # Takes a tuple of (left, right) target pairs and finds routines that could hit the ball between those target pairs
    # Only meant for routines that require a defined intercept time/place in the future

    # Here we get the slices that need to be searched - by defining a cap, we can reduce the number of slices and improve search times
    slices = get_slices(agent, cap_)

    if slices is None:
        return

    # Construct the target
    options = rlru.TargetOptions(*slices)
    target_id = rlru.new_target(tuple(target[0]), tuple(target[1]), agent.index, options)

    # Search for the shot
    return _get_shot_with_target_id(target_id, can_ground, can_jump, can_double_jump, can_aerial)


def find_any_shot(agent: VirxERLU, cap_: int=6, can_aerial: bool=True, can_double_jump: bool=True, can_jump: bool=True, can_ground: bool=True) -> Optional[Union[GroundShot, JumpShot, DoubleJumpShot, AerialShot]]:
    if not can_aerial and not can_double_jump and not can_jump and not can_ground:
        agent.print("WARNING: All shots were disabled when find_shot was ran")
        return

    # Only meant for routines that require a defined intercept time/place in the future

    # Here we get the slices that need to be searched - by defining a cap, we can reduce the number of slices and improve search times
    slices = get_slices(agent, cap_)

    if slices is None:
        return

    # Construct the target
    options = rlru.TargetOptions(*slices)
    target_id = rlru.new_any_target(agent.index, options)

    # Search for the shot
    return _get_shot_with_target_id(target_id, can_ground, can_jump, can_double_jump, can_aerial)


def _get_shot_with_target_id(target_id: int, may_ground_shot: bool, may_jump_shot: bool, may_double_jump_shot: bool, may_aerial_shot: bool) -> Optional[Union[GroundShot, JumpShot, DoubleJumpShot, AerialShot]]:
    shot = rlru.get_shot_with_target(target_id, False, may_ground_shot, may_jump_shot, may_double_jump_shot, may_aerial_shot, only=True)

    if shot.found:
        return SHOT_SWITCH[int(shot.shot_type)](shot.time, target_id, shot.is_forwards, Vector(*shot.shot_vector))


def get_slices(agent: VirxERLU, cap_: int) -> Optional[Tuple[int, int]]:
    start_slice = 0
    end_slices = None

    # If we're shooting, crop the struct
    if agent.shooting and agent.stack[0].__class__.__name__ != "short_shot":
        # Get the time remaining
        time_remaining = agent.stack[0].intercept_time - agent.time

        # if the shot is done but it's working on it's 'follow through', then ignore this stuff
        if time_remaining > 0:
            # Convert the time remaining into number of slices, and take off the minimum gain accepted from the time
            min_gain = 0.2
            end_slice = round(min(time_remaining - min_gain, cap_) * 120)

    if end_slices is None:
        # Cap the slices
        end_slice = round(cap_ * 120)

    # We can't end a slice index that's lower than the start index
    if end_slice <= start_slice:
        return

    return start_slice, end_slice
