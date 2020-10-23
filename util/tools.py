from util.routines import Aerial, double_jump, ground_shot, jump_shot, virxrlcu
from util.utils import Vector, math, side


def find_ground_shot(agent, target, weight=None, cap_=6):
    return find_shot(agent, target, weight, cap_, can_aerial=False, can_double_jump=False, can_jump=False)


def find_any_ground_shot(agent, cap_=6):
    return find_any_shot(agent, cap_, can_aerial=False, can_double_jump=False, can_jump=False)


def find_jump_shot(agent, target, weight=None, cap_=6):
    return find_shot(agent, target, weight, cap_, can_aerial=False, can_double_jump=False, can_ground=False)


def find_any_jump_shot(agent, cap_=6):
    return find_any_shot(agent, cap_, can_aerial=False, can_double_jump=False, can_ground=False)


def find_double_jump(agent, target, weight=None, cap_=6):
    return find_shot(agent, target, weight, cap_, can_aerial=False, can_jump=False, can_ground=False)


def find_any_double_jump(agent, cap_=6):
    return find_any_shot(agent, cap_, can_aerial=False, can_jump=False, can_ground=False)


def find_aerial(agent, target, weight=None, cap_=6):
    return find_shot(agent, target, weight, cap_, can_double_jump=False, can_jump=False, can_ground=False)


def find_any_aerial(agent, cap_=6):
    return find_any_shot(agent, cap_, can_double_jump=False, can_jump=False, can_ground=False)


def find_shot(agent, target, weight=None, cap_=6, can_aerial=True, can_double_jump=True, can_jump=True, can_ground=True):
    if not can_aerial and not can_double_jump and not can_jump and not can_ground:
        agent.print("WARNING: All shots were disabled when find_shot was ran")
        return

    # Takes a tuple of (left,right) target pairs and finds routines that could hit the ball between those target pairs
    # Only meant for routines that require a defined intercept time/place in the future
    # Here we get the slices that need to be searched - by defining a cap, we can reduce the number of slices and improve search times
    slices = get_slices(agent, cap_, weight=weight)

    if slices is None:
        return

    # Assemble data in a form that can be passed to CPython
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

    if can_aerial or can_ground:
        me_a = (
            me[0],
            agent.me.velocity.tuple(),
            agent.me.up.tuple(),
            me[1],
            agent.me.hitbox.tuple(),
            1 if agent.me.airborne else -1,
            me[2]
        )

        gravity = agent.gravity.tuple()

        max_aerial_height = 1200 if len(agent.friends) == 0 and len(agent.foes) == 1 else math.inf
        min_aerial_height = 551 if max_aerial_height > 1200 and agent.me.location.z >= 2044 - agent.me.hitbox.height * 1.1 else (150 if agent.boost_amount == 'unlimited' or agent.me.airborne else 450)

    is_on_ground = not agent.me.airborne

    # Loop through the slices
    for ball_slice in slices:
        # Gather some data about the slice
        intercept_time = ball_slice.game_seconds
        time_remaining = intercept_time - agent.time

        if time_remaining <= 0:
            return

        ball_location = (ball_slice.physics.location.x, ball_slice.physics.location.y, ball_slice.physics.location.z)

        if abs(ball_location[1]) > 5212.75:
            return  # abandon search if ball is scored at/after this point

        # Check if we can make a shot at this slice
        # This operation is very expensive, so we use CPython to improve run time
        if is_on_ground:
            if can_ground and ball_location[2]:
                shot = virxrlcu.parse_slice_for_ground_shot_with_target(time_remaining, *game_info, ball_location, me_a, *target)

                # If we found a viable shot, pass the data into the shot routine and return the shot
                if shot['found'] == 1:
                    return ground_shot(intercept_time, (Vector(*shot['targets'][0]), Vector(*shot['targets'][1])))

            if time_remaining >= 0.5:
                if can_jump and ball_location[2] >= 92.75 + agent.me.hitbox.height:
                    shot = virxrlcu.parse_slice_for_jump_shot_with_target(time_remaining - 0.3, *game_info, ball_location, *me, *target)

                    # If we found a viable shot, pass the data into the shot routine and return the shot
                    if shot['found'] == 1:
                        return jump_shot(intercept_time, (Vector(*shot['targets'][0]), Vector(*shot['targets'][1])))

                if can_double_jump:
                    shot = virxrlcu.parse_slice_for_double_jump_with_target(time_remaining - 0.3, *game_info, ball_location, *me, *target)

                    # If we found a viable shot, pass the data into the shot routine and return the shot
                    if shot['found'] == 1:
                        return double_jump(intercept_time, (Vector(*shot['targets'][0]), Vector(*shot['targets'][1])))

        if can_aerial and not (min_aerial_height > ball_location[2] or ball_location[2] > max_aerial_height):
            shot = virxrlcu.parse_slice_for_aerial_shot_with_target(time_remaining, *game_info, gravity, ball_location, me_a, *target)

            # If we found a viable shot, pass the data into the shot routine and return the shot
            if shot['found'] == 1:
                return Aerial(intercept_time, (Vector(*shot['targets'][0]), Vector(*shot['targets'][1])), shot['fast'])


def find_any_shot(agent, cap_=6, can_aerial=True, can_double_jump=True, can_jump=True, can_ground=True):
    if not can_aerial and not can_double_jump and not can_jump and not can_ground:
        agent.print("WARNING: All shots were disabled when find_shot was ran")
        return

    # Only meant for routines that require a defined intercept time/place in the future
    # Here we get the slices that need to be searched - by defining a cap, we can reduce the number of slices and improve search times
    slices = get_slices(agent, cap_)

    if slices is None:
        return

    # Assemble data in a form that can be passed to CPython
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

    if can_aerial or can_ground:
        me_a = (
            me[0],
            agent.me.velocity.tuple(),
            agent.me.up.tuple(),
            me[1],
            agent.me.hitbox.tuple(),
            1 if agent.me.airborne else -1,
            me[2]
        )

        gravity = agent.gravity.tuple()

        max_aerial_height = 1200 if len(agent.friends) == 0 and len(agent.foes) == 1 else math.inf
        min_aerial_height = 551 if max_aerial_height > 1200 and agent.me.location.z >= 2044 - agent.me.hitbox.height * 1.1 else (150 if agent.boost_amount == 'unlimited' or agent.me.airborne else 450)

    is_on_ground = not agent.me.airborne

    # Loop through the slices
    for ball_slice in slices:
        # Gather some data about the slice
        intercept_time = ball_slice.game_seconds
        time_remaining = intercept_time - agent.time

        if time_remaining <= 0:
            return

        ball_location = (ball_slice.physics.location.x, ball_slice.physics.location.y, ball_slice.physics.location.z)

        if abs(ball_location[1]) > 5212.75:
            return  # abandon search if ball is scored at/after this point

        # Check if we can make a shot at this slice
        # This operation is very expensive, so we use CPython to improve run time
        if is_on_ground:
            if can_ground and ball_location[2] < 92.75 + agent.me.hitbox.height:
                shot = virxrlcu.parse_slice_for_ground_shot(time_remaining, *game_info, ball_location, me_a)

                # If we found a viable shot, pass the data into the shot routine and return the shot
                if shot['found'] == 1:
                    return ground_shot(intercept_time)

            if time_remaining >= 0.5:
                if can_jump and ball_location[2] >= 92.75 + agent.me.hitbox.height:
                    shot = virxrlcu.parse_slice_for_jump_shot(time_remaining - 0.3, *game_info, ball_location, *me)

                    # If we found a viable shot, pass the data into the shot routine and return the shot
                    if shot['found'] == 1:
                        return jump_shot(intercept_time)

                if can_double_jump:
                    shot = virxrlcu.parse_slice_for_double_jump(time_remaining - 0.3, *game_info, ball_location, *me)

                    # If we found a viable shot, pass the data into the shot routine and return the shot
                    if shot['found'] == 1:
                        return double_jump(intercept_time)

        if can_aerial and not (min_aerial_height > ball_location[2] or ball_location[2] > max_aerial_height):
            shot = virxrlcu.parse_slice_for_aerial_shot(time_remaining, *game_info, gravity, ball_location, me_a)

            # If we found a viable shot, pass the data into the shot routine and return the shot
            if shot['found'] == 1:
                return Aerial(intercept_time, fast_aerial=shot['fast'])


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

        if time_remaining <= 3:
            # Half the time, double the slices
            slices = struct.slices[start_slice:end_slice]
        else:
            # Skip every other slice (for performance reasons)
            slices = struct.slices[start_slice:end_slice:2]
    else:
        # Cap the slices
        end_slice = math.ceil(cap_ * 60)

        # Start 0.2 seconds in, and skip every other slice
        slices = struct.slices[start_slice:end_slice:2]

    # get the number of slices
    s_len = len(slices)

    # find what's exactly 1 quarter of the total slices
    quart = s_len / 4

    # get the integer 1 quarter and 3 quarter slices
    main_s = (math.ceil(quart), math.floor(quart * 3))

    # search the middle half, then the first quarter, then the last quarter
    main = slices[main_s[0]:main_s[1]]
    main += slices[:main_s[0]]
    main += slices[main_s[1]:]

    return main
