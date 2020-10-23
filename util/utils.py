from queue import Full

from util.agent import Vector, math


def backsolve(target, car, time, gravity):
    # Finds the acceleration required for a car to reach a target in a specific amount of time
    d = target - car.location
    dvx = ((d.x/time) - car.velocity.x) / time
    dvy = ((d.y/time) - car.velocity.y) / time
    dvz = (((d.z/time) - car.velocity.z) / time) - (gravity.z * time)
    return Vector(dvx, dvy, dvz)


def cap(x, low, high):
    # caps/clamps a number between a low and high value
    return max(min(x, high), low)


def defaultPD(agent, local_target, upside_down=False, up=None):
    # points the car towards a given local target.
    # Direction can be changed to allow the car to steer towards a target while driving backwards

    if up is None:
        up = agent.me.local(Vector(z=-1 if upside_down else 1))  # where "up" is in local coordinates
    target_angles = (
        math.atan2(local_target.z, local_target.x),  # angle required to pitch towards target
        math.atan2(local_target.y, local_target.x),  # angle required to yaw towards target
        math.atan2(up.y, up.z)  # angle required to roll upright
    )
    # Once we have the angles we need to rotate, we feed them into PD loops to determing the controller inputs
    agent.controller.steer = steerPD(target_angles[1], 0)
    agent.controller.pitch = steerPD(target_angles[0], agent.me.angular_velocity.y/4)
    agent.controller.yaw = steerPD(target_angles[1], -agent.me.angular_velocity.z/4)
    agent.controller.roll = steerPD(target_angles[2], agent.me.angular_velocity.x/2)
    # Returns the angles, which can be useful for other purposes
    return target_angles


def defaultThrottle(agent, target_speed, target_angles=None):
    # accelerates the car to a desired speed using throttle and boost
    car_speed = agent.me.local_velocity().x
    t = target_speed - car_speed

    if not agent.me.airborne:
        angle_to_target = abs(target_angles[1])
        if target_angles is not None:
            agent.controller.handbrake = not agent.me.airborne and ((angle_to_target > 1.54) if sign(car_speed) == 1 else (angle_to_target < 1.6)) and agent.me.velocity.magnitude() > 150

        if agent.controller.handbrake:
            if car_speed < 900:
                agent.controller.throttle = sign(t)
                agent.controller.steer = sign(agent.controller.steer)
            else:
                agent.controller.throttle = -sign(car_speed)
                agent.controller.handbrake = False
        else:
            agent.controller.throttle = cap((t**2) * sign(t)/1000, -1, 1)
            agent.controller.boost = (t > 150 or (target_speed > 1400 and t > agent.boost_accel / 30)) and agent.controller.throttle > 0.9 and angle_to_target < 0.5

    return car_speed


def defaultDrive(agent, target_speed, local_target):
    if target_speed < 0:
        local_target *= -1

    target_angles = defaultPD(agent, local_target)
    velocity = defaultThrottle(agent, target_speed, target_angles)

    return target_angles, velocity


def in_field(point, radius):
    # determines if a point is inside the standard soccer field
    point = Vector(abs(point.x), abs(point.y), abs(point.z))
    return not (point.x > 4080 - radius or point.y > 5900 - radius or (point.x > 880 - radius and point.y > 5105 - radius) or (point.x > 2650 and point.y > -point.x + 8025 - radius))


def find_slope(shot_vector, car_to_target):
    # Finds the slope of your car's position relative to the shot vector (shot vector is y axis)
    # 10 = you are on the axis and the ball is between you and the direction to shoot in
    # -10 = you are on the wrong side
    # 1 = you're about 45 degrees offcenter
    d = shot_vector.dot(car_to_target)
    e = abs(shot_vector.cross(Vector(z=1)).dot(car_to_target))
    try:
        f = d / e
    except ZeroDivisionError:
        return 10*sign(d)
    return cap(f, -3, 3)


def post_correction(ball_location, left_target, right_target):
    # this function returns target locations that are corrected to account for the ball's radius
    # If the left and right post swap sides, a goal cannot be scored
    # We purposely make this a bit larger so that our shots have a higher chance of success
    ball_radius = 120
    goal_line_perp = (right_target - left_target).cross(Vector(z=1))
    left = left_target + ((left_target - ball_location).normalize().cross(Vector(z=-1))*ball_radius)
    right = right_target + ((right_target - ball_location).normalize().cross(Vector(z=1))*ball_radius)
    left = left_target if (left-left_target).dot(goal_line_perp) > 0 else left
    right = right_target if (right-right_target).dot(goal_line_perp) > 0 else right
    swapped = (left - ball_location).normalize().cross(Vector(z=1)).dot((right - ball_location).normalize()) > -0.1
    return left, right, swapped


def quadratic(a, b, c):
    # Returns the two roots of a quadratic
    inside = (b*b) - (4*a*c)

    try:
        inside = math.sqrt(inside)
    except ValueError:
        return -1, -1

    if a == 0:
        return -1, -1

    b = -b
    a = 2*a

    return (b + inside)/a, (b - inside)/a


def side(x):
    # returns -1 for blue team and 1 for orange team
    return (-1, 1)[x]


def sign(x):
    # returns the sign of a number, -1, 0, +1
    if x < 0:
        return -1

    if x > 0:
        return 1

    return 0


def steerPD(angle, rate):
    # A Proportional-Derivative control loop used for defaultPD
    return cap(((35*(angle+rate))**3)/10, -1, 1)


def lerp(a, b, t):
    # Linearly interpolate from a to b using t
    # For instance, when t == 0, a is returned, and when t is 1, b is returned
    # Works for both numbers and Vectors
    return (b - a) * t + a


def invlerp(a, b, v):
    # Inverse linear interpolation from a to b with value v
    # For instance, it returns 0 if v is a, and returns 1 if v is b, and returns 0.5 if v is exactly between a and b
    # Works for both numbers and Vectors
    return (v - a) / (b - a)


def send_comm(agent, msg):
    message = {
        "index": agent.index,
        "team": agent.team
    }
    msg.update(message)
    try:
        agent.matchcomms.outgoing_broadcast.put_nowait({
            "VirxERLU": msg
        })
    except Full:
        agent.print("Outgoing broadcast is full; couldn't send message")


def peek_generator(generator):
    try:
        return next(generator)
    except StopIteration:
        return


def almost_equals(x, y, threshold):
    return x - threshold < y and y < x + threshold
