from typing import Optional, Tuple

import virx_erlu_rlib as rlru

from util.agent import VirxERLU
from util.routines import GroundShot, JumpShot
from util.utils import Vector

SHOT_SWITCH = {
    rlru.ShotType.GROUND: GroundShot,
    rlru.ShotType.JUMP: JumpShot,
    # ShotType.DOUBLE_JUMP: double_jump
}


def find_ground_shot(agent, target, cap_=6):
    return find_shot(agent, target, cap_, can_aerial=False, can_double_jump=False, can_jump=False)


# def find_any_ground_shot(agent, cap_=6):
#     return find_any_shot(agent, cap_, can_aerial=False, can_double_jump=False, can_jump=False)


# def find_jump_shot(agent, target, cap_=6):
#     return find_shot(agent, target, cap_, can_aerial=False, can_double_jump=False, can_ground=False)


# def find_any_jump_shot(agent, cap_=6):
#     return find_any_shot(agent, cap_, can_aerial=False, can_double_jump=False, can_ground=False)


# def find_double_jump(agent, target, cap_=6):
#     return find_shot(agent, target, cap_, can_aerial=False, can_jump=False, can_ground=False)


# def find_any_double_jump(agent, cap_=6):
#     return find_any_shot(agent, cap_, can_aerial=False, can_jump=False, can_ground=False)


# def find_aerial(agent, target, cap_=6):
#     return find_shot(agent, target, cap_, can_double_jump=False, can_jump=False, can_ground=False)


# def find_any_aerial(agent, cap_=6):
#     return find_any_shot(agent, cap_, can_double_jump=False, can_jump=False, can_ground=False)


def find_shot(agent: VirxERLU, target: Tuple[Vector, Vector], cap_: int=6, can_aerial: bool=True, can_double_jump: bool=True, can_jump: bool=True, can_ground: bool=True):
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
    target_id = rlru.new_target(tuple(target[0]), tuple(target[1]), agent.me.index, options)

    # Search for the shot
    shot = rlru.get_shot_with_target(target_id, may_ground_shot=can_ground, only=True)

    if shot.found:
        return SHOT_SWITCH[shot.shot_type](shot.time, target_id)


# def find_any_shot(agent, cap_=6, can_aerial=True, can_double_jump=True, can_jump=True, can_ground=True):
#     if not can_aerial and not can_double_jump and not can_jump and not can_ground:
#         agent.print("WARNING: All shots were disabled when find_any_shot was ran")
#         return

#     # Only meant for routines that require a defined intercept time/place in the future

#     # Assemble data in a form that can be passed to C
#     me = agent.me.get_raw(agent)

#     game_info = (
#         agent.boost_accel,
#         agent.ball_radius
#     )

#     gravity = tuple(agent.gravity)

#     is_on_ground = not agent.me.airborne
#     can_ground = is_on_ground and can_ground
#     can_jump = is_on_ground and can_jump
#     can_double_jump = is_on_ground and can_double_jump
#     can_aerial = (not is_on_ground or agent.time - agent.me.land_time > 0.5) and can_aerial
#     any_ground = can_ground or can_jump or can_double_jump

#     if not any_ground and not can_aerial:
#         return

#     # Here we get the slices that need to be searched - by defining a cap, we can reduce the number of slices and improve search times
#     slices = get_slices(agent, cap_)

#     if slices is None:
#         return

#     # Loop through the slices
#     for ball_slice in slices:
#         # Gather some data about the slice
#         intercept_time = ball_slice.game_seconds
#         T = intercept_time - agent.time - (1 / 120)

#         if T <= 0:
#             return

#         ball_location = (ball_slice.physics.location.x, ball_slice.physics.location.y, ball_slice.physics.location.z)

#         if abs(ball_location[1]) > 5212.75:
#             return  # abandon search if ball is scored at/after this point

#         ball_info = (ball_location, (ball_slice.physics.velocity.x, ball_slice.physics.velocity.y, ball_slice.physics.velocity.z))

#         # Check if we can make a shot at this slice
#         # This operation is very expensive, so we use C to improve run time
#         shot = virxrlcu.parse_slice_for_shot(can_ground, can_jump, can_double_jump, can_aerial, T, *game_info, gravity, ball_info, me)

#         if shot['found'] == 1:
#             shot_type = ShotType(shot["shot_type"])
#             if shot_type == ShotType.AERIAL:
#                 return Aerial(intercept_time, fast_aerial=shot['fast'])

#             return SHOT_SWITCH[shot_type](intercept_time)


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
