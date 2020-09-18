from util.routines import virxrlcu, Aerial, jump_shot, double_jump
from util.utils import Vector, math, side


def find_jump_shot(agent, target, weight=None, cap_=6):
    # Takes a tuple of (left,right) target pairs and finds routines that could hit the ball between those target pairs
    # Only meant for routines that require a defined intercept time/place in the future
    # Here we get the slices that need to be searched - by defining a cap or a weight, we can reduce the number of slices and improve search times
    slices = get_slices(agent, cap_, weight=weight)

    if slices is None:
        return

    # Assemble data in a form that can be passed to C
    target = (
        target[0].tuple(),
        target[1].tuple()
    )

    me = (
        agent.me.location.tuple(),
        agent.me.forward.tuple(),
        agent.me.boost if agent.boost_amount != 'unlimited' else 100000,
        agent.me.local_velocity().x
    )

    game_info = (
        agent.best_shot_value,
        agent.boost_accel
    )

    # Loop through the slices
    for ball_slice in slices:
        # Gather some data about the slice
        intercept_time = ball_slice.game_seconds
        time_remaining = intercept_time - agent.time

        if time_remaining <= 0:
            continue

        ball_location = (ball_slice.physics.location.x, ball_slice.physics.location.y, ball_slice.physics.location.z)

        if abs(ball_location[1]) > 5212:
            return  # abandon search if ball is scored at/after this point

        # Check if we can make a shot at this slice
        # This operation is very expensive, so we use a custom C function to improve run time
        shot = virxrlcu.parse_slice_for_jump_shot_with_target(time_remaining - 0.1, *game_info, ball_location, *me, *target)

        # If we found a viable shot, pass the data into the shot routine and return the shot
        if shot['found'] == 1:
            return jump_shot(intercept_time, Vector(*shot['best_shot_vector']))


def find_any_jump_shot(agent, cap_=3):
    slices = get_slices(agent, cap_)

    if slices is None:
        return

    me = (
        agent.me.location.tuple(),
        agent.me.forward.tuple(),
        agent.me.boost if agent.boost_amount != 'unlimited' else 1000000,
        agent.me.local_velocity().x
    )

    game_info = (
        agent.best_shot_value,
        agent.boost_accel
    )

    for ball_slice in slices:
        intercept_time = ball_slice.game_seconds
        time_remaining = intercept_time - agent.time

        if time_remaining <= 0:
            continue

        ball_location = (ball_slice.physics.location.x, ball_slice.physics.location.y, ball_slice.physics.location.z)

        if abs(ball_location[1]) > 5212:
            return

        shot = virxrlcu.parse_slice_for_jump_shot(time_remaining - 0.1, *game_info, ball_location, *me)

        if shot['found'] == 1:
            return jump_shot(intercept_time, Vector(*shot['best_shot_vector']))


def find_double_jump(agent, target, weight=None, cap_=6):
    slices = get_slices(agent, cap_, weight=weight, start_slice=30)

    if slices is None:
        return

    target = (
        target[0].tuple(),
        target[1].tuple()
    )

    me = (
        agent.me.location.tuple(),
        agent.me.forward.tuple(),
        agent.me.boost if agent.boost_amount != 'unlimited' else 100000,
        agent.me.local_velocity().x
    )

    game_info = (
        agent.best_shot_value,
        agent.boost_accel
    )

    for ball_slice in slices:
        intercept_time = ball_slice.game_seconds
        time_remaining = intercept_time - agent.time

        if time_remaining <= 0:
            continue

        ball_location = (ball_slice.physics.location.x, ball_slice.physics.location.y, ball_slice.physics.location.z)

        if abs(ball_location[1]) > 5212:
            return

        shot = virxrlcu.parse_slice_for_double_jump_with_target(time_remaining - 0.3, *game_info, ball_location, *me, *target)

        if shot['found'] == 1:
            return double_jump(intercept_time, Vector(*shot['best_shot_vector']))


def find_any_double_jump(agent, cap_=3):
    slices = get_slices(agent, cap_, start_slice=30)

    if slices is None:
        return

    me = (
        agent.me.location.tuple(),
        agent.me.forward.tuple(),
        agent.me.boost if agent.boost_amount != 'unlimited' else 100000,
        agent.me.local_velocity().x
    )

    game_info = (
        agent.best_shot_value,
        agent.boost_accel
    )

    for ball_slice in slices:
        intercept_time = ball_slice.game_seconds
        time_remaining = intercept_time - agent.time

        if time_remaining <= 0:
            continue

        ball_location = (ball_slice.physics.location.x, ball_slice.physics.location.y, ball_slice.physics.location.z)

        if abs(ball_location[1]) > 5212:
            return

        shot = virxrlcu.parse_slice_for_double_jump(time_remaining - 0.3, agent.best_shot_value, ball_location, *me)

        if shot['found'] == 1:
            return double_jump(intercept_time, Vector(*shot['best_shot_vector']))


def find_aerial(agent, target, weight=None, cap_=6):
    slices = get_slices(agent, cap_, weight=weight)

    if slices is None:
        return

    target = (
        target[0].tuple(),
        target[1].tuple()
    )

    me = (
        agent.me.location.tuple(),
        agent.me.velocity.tuple(),
        agent.me.up.tuple(),
        agent.me.forward.tuple(),
        1 if agent.me.airborne else -1,
        agent.me.boost if agent.boost_amount != 'unlimited' else 100000
    )

    gravity = agent.gravity.tuple()

    max_aerial_height = 643 if len(agent.friends) == 0 and len(agent.foes) == 1 else math.inf
    min_aerial_height = 551 if max_aerial_height > 643 and agent.me.location.z >= 2044 - agent.me.hitbox.height * 1.1 else (0 if agent.boost_amount == 'unlimited' else 450)

    for ball_slice in slices:
        intercept_time = ball_slice.game_seconds
        time_remaining = intercept_time - agent.time

        if time_remaining <= 0:
            return

        ball_location = (ball_slice.physics.location.x, ball_slice.physics.location.y, ball_slice.physics.location.z)

        if abs(ball_location[1]) > 5212:
            return

        if min_aerial_height > ball_location[2] or ball_location[2] > max_aerial_height:
            continue

        shot = virxrlcu.parse_slice_for_aerial_shot_with_target(time_remaining, agent.best_shot_value, agent.boost_accel, gravity, ball_location, me, *target)
        if shot['found'] == 1:
            return Aerial(intercept_time, Vector(*shot['best_shot_vector']), shot['fast'])


def find_any_aerial(agent, cap_=3):
    slices = get_slices(agent, cap_)

    if slices is None:
        return

    me = (
        agent.me.location.tuple(),
        agent.me.velocity.tuple(),
        agent.me.up.tuple(),
        agent.me.forward.tuple(),
        1 if agent.me.airborne else -1,
        agent.me.boost if agent.boost_amount != 'unlimited' else 100000
    )

    gravity = agent.gravity.tuple()

    max_aerial_height = 735 if len(agent.friends) == 0 and len(agent.foes) == 1 else math.inf
    min_aerial_height = 551 if max_aerial_height > 643 and agent.me.location.z >= 2044 - agent.me.hitbox.height * 1.1 else (0 if agent.boost_amount == 'unlimited' else 450)

    for ball_slice in slices:
        intercept_time = ball_slice.game_seconds
        time_remaining = intercept_time - agent.time

        if time_remaining <= 0:
            return

        ball_location = (ball_slice.physics.location.x, ball_slice.physics.location.y, ball_slice.physics.location.z)

        if abs(ball_location[1]) > 5212:
            return

        if min_aerial_height > ball_location[2] or ball_location[2] > max_aerial_height:
            continue

        shot = virxrlcu.parse_slice_for_aerial_shot(time_remaining, agent.best_shot_value, agent.boost_accel, gravity, ball_location, me)

        if shot['found'] == 1:
            return Aerial(intercept_time, Vector(*shot['best_shot_vector']), shot['fast'])


def find_shot(agent, target, weight=None, cap_=6, can_aerial=True, can_double_jump=True, can_jump=True):
    if not can_aerial and not can_double_jump and not can_jump:
        agent.print("WARNING: All shots were disabled when find_shot was ran")
        return

    slices = get_slices(agent, cap_, weight=weight)

    if slices is None:
        return

    target = (
        target[0].tuple(),
        target[1].tuple()
    )

    me = (
        agent.me.location.tuple(),
        agent.me.forward.tuple(),
        agent.me.boost if agent.boost_amount != 'unlimited' else 100000,
        agent.me.local_velocity().x
    )

    game_info = (
        agent.best_shot_value,
        agent.boost_accel
    )

    if can_aerial:
        me_a = (
            me[0],
            agent.me.velocity.tuple(),
            agent.me.up.tuple(),
            me[1],
            1 if agent.me.airborne else -1,
            me[2]
        )

        gravity = agent.gravity.tuple()

        max_aerial_height = 643 if len(agent.friends) == 0 and len(agent.foes) == 1 else math.inf
        min_aerial_height = 551 if max_aerial_height > 643 and agent.me.location.z >= 2044 - agent.me.hitbox.height * 1.1 else (0 if agent.boost_amount == 'unlimited' else 450)

    for ball_slice in slices:
        intercept_time = ball_slice.game_seconds
        time_remaining = intercept_time - agent.time

        if time_remaining <= 0:
            return

        ball_location = (ball_slice.physics.location.x, ball_slice.physics.location.y, ball_slice.physics.location.z)

        if abs(ball_location[1]) > 5212:
            return

        if can_jump:
            shot = virxrlcu.parse_slice_for_jump_shot_with_target(time_remaining - 0.1, *game_info, ball_location, *me, *target)

            if shot['found'] == 1:
                return jump_shot(intercept_time, Vector(*shot['best_shot_vector']))
        
        if can_double_jump and time_remaining >= 0.5:
            shot = virxrlcu.parse_slice_for_double_jump_with_target(time_remaining - 0.3, *game_info, ball_location, *me, *target)

            if shot['found'] == 1:
                return double_jump(intercept_time, Vector(*shot['best_shot_vector']))

        if can_aerial and not (min_aerial_height > ball_location[2] or ball_location[2] > max_aerial_height):
            shot = virxrlcu.parse_slice_for_aerial_shot_with_target(time_remaining, *game_info, gravity, ball_location, me_a, *target)

            if shot['found'] == 1:
                return Aerial(intercept_time, Vector(*shot['best_shot_vector']), shot['fast'])


def find_any_shot(agent, cap_=3, can_aerial=True, can_double_jump=True, can_jump=True):
    if not can_aerial and not can_double_jump and not can_jump:
        agent.print("WARNING: All shots were disabled when find_shot was ran")
        return

    slices = get_slices(agent, cap_)

    if slices is None:
        return

    me = (
        agent.me.location.tuple(),
        agent.me.forward.tuple(),
        agent.me.boost if agent.boost_amount != 'unlimited' else 100000,
        agent.me.local_velocity().x
    )

    game_info = (
        agent.best_shot_value,
        agent.boost_accel
    )

    if can_aerial:
        me_a = (
            me[0],
            agent.me.velocity.tuple(),
            agent.me.up.tuple(),
            me[1],
            1 if agent.me.airborne else -1,
            me[2]
        )

        gravity = agent.gravity.tuple()

        max_aerial_height = 643 if len(agent.friends) == 0 and len(agent.foes) == 1 else math.inf
        min_aerial_height = 551 if max_aerial_height > 643 and agent.me.location.z >= 2044 - agent.me.hitbox.height * 1.1 else (0 if agent.boost_amount == 'unlimited' else 450)

    for ball_slice in slices:
        intercept_time = ball_slice.game_seconds
        time_remaining = intercept_time - agent.time

        if time_remaining <= 0:
            return

        ball_location = (ball_slice.physics.location.x, ball_slice.physics.location.y, ball_slice.physics.location.z)

        if abs(ball_location[1]) > 5212:
            return

        if can_jump:
            shot = virxrlcu.parse_slice_for_jump_shot(time_remaining - 0.1, *game_info, ball_location, *me)

            if shot['found'] == 1:
                return jump_shot(intercept_time, Vector(*shot['best_shot_vector']))
        
        if can_double_jump and time_remaining >= 0.5:
            shot = virxrlcu.parse_slice_for_double_jump(time_remaining - 0.3, *game_info, ball_location, *me)

            if shot['found'] == 1:
                return double_jump(intercept_time, Vector(*shot['best_shot_vector']))

        if can_aerial and not (min_aerial_height > ball_location[2] or ball_location[2] > max_aerial_height):
            shot = virxrlcu.parse_slice_for_aerial_shot(time_remaining, *game_info, gravity, ball_location, me_a)

            if shot['found'] == 1:
                return Aerial(intercept_time, Vector(*shot['best_shot_vector']), shot['fast'])


def get_slices(agent, cap_, weight=None, start_slice=12):
    # Get the struct
    struct = agent.ball_prediction_struct

    # Make sure it isn't empty
    if struct is None:
        return

    ball_y = agent.ball.location.y * side(agent.team)
    foes = tuple(foe for foe in agent.foes if not foe.demolished and foe.location.y * side(agent.team) < ball_y)

    # If we're shooting, crop the struct
    if agent.shooting:
        # Get the time remaining
        time_remaining = agent.stack[0].intercept_time - agent.time

        # Convert the time remaining into number of slices, and take off the minimum gain accepted from the time
        min_gain = 0.05
        end_slice = math.ceil(min(time_remaining - min_gain, cap_) * 60)

        # We can't end a slice index that's lower than the start index
        if end_slice <= 12:
            return

        # Half the time, double the slices
        if time_remaining <= 3:
            return struct.slices[start_slice:end_slice]

        return struct.slices[start_slice:end_slice:2]

    # If we're not shooting, then cap the slices at the cap
    end_slice = math.ceil(cap_ * 60)

    # Start 0.2 seconds in, and skip every other data point
    return struct.slices[start_slice:end_slice:2]
