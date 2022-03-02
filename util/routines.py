from __future__ import annotations

import math
from traceback import print_exc
from typing import Optional

import virx_erlu_rlib as rlru

from util import utils
from util.agent import BaseRoutine, Vector, VirxERLU, boost_object, rlru


class GroundShot(BaseRoutine):
    def __init__(self, intercept_time: float, target_id: int):
        self.intercept_time = intercept_time
        self.target_id = target_id

    def update(self, shot: GroundShot):
        try:
            rlru.confirm_target(shot.target_id)
        except Exception:
            print_exc()
            return
        rlru.remove_target(self.target_id)
        self.intercept_time = shot.intercept_time
        self.target_id = shot.target_id

    def run(self, agent: VirxERLU):
        future_ball_location = Vector(*rlru.get_slice(self.intercept_time).location)

        agent.point(future_ball_location, agent.renderer.purple())

        T = self.intercept_time - agent.time

        try:
            shot_info = rlru.get_data_for_shot_with_target(self.target_id)
        except AssertionError:
            agent.pop()
            print_exc()
            return
        except IndexError:
            agent.pop()
            print_exc()
            return
        except ValueError:
            # We ran out of time
            agent.pop()
            return

        final_target = Vector(*shot_info.final_target)
        agent.point(final_target, agent.renderer.red())

        if len(shot_info.path_samples) > 2:
            agent.polyline(tuple(Vector(sample[0], sample[1], 30) for sample in shot_info.path_samples), agent.renderer.lime())
        else:
            agent.line(agent.me.location, final_target, agent.renderer.lime())

        distance_remaining = shot_info.distance_remaining
        speed_required = min(distance_remaining / T, 2300)
        local_final_target = agent.me.local_location(final_target.flatten())

        utils.defaultDrive(agent, speed_required, local_final_target)

    def on_push(self):
        rlru.confirm_target(self.target_id)

    def pre_pop(self):
        rlru.remove_target(self.target_id)
ground_shot = GroundShot  # legacy


class Recovery(BaseRoutine):
    # Point towards our velocity vector and land upright, unless we aren't moving very fast
    # A vector can be provided to control where the car points when it lands
    def __init__(self, target: Optional[Vector]=None):
        self.target = target

    def run(self, agent: VirxERLU):
        target = agent.me.velocity.normalize() if self.target is None else (self.target - agent.me.location).normalize()

        landing_plane = utils.find_landing_plane(agent.me.location, agent.me.velocity, agent.gravity.z)

        d_switch = [
            "side wall",
            "side wall",
            "back wall",
            "back wall",
            "ceiling",
            "floor"
        ]

        agent.dbg_2d(f"Recovering towards the {d_switch[landing_plane]}")

        t_switch = [
            Vector(y=target.y, z=-1),
            Vector(y=target.y, z=-1),
            Vector(x=target.x, z=-1),
            Vector(x=target.x, z=-1),
            Vector(x=target.x, y=target.y),
            Vector(x=target.x, y=target.y)
        ]

        r_switch = [
            Vector(x=-1),
            Vector(x=1),
            Vector(y=-1),
            Vector(y=1),
            Vector(z=-1),
            Vector(z=1)
        ]

        utils.defaultPD(agent, agent.me.local(t_switch[landing_plane]), up=agent.me.local(r_switch[landing_plane]))
        agent.controller.throttle = 1
        if not agent.me.airborne:
            agent.pop()
recovery = Recovery  # legacy


def brake_handler(agent: VirxERLU) -> bool:
    # current forward velocity
    speed = agent.me.local_velocity().x
    # apply our throttle in the opposite direction
    agent.controller.throttle = -utils.cap(speed / utils.BRAKE_ACC, -1, 1)
    # threshold of "we're close enough to stopping"
    return abs(speed) > 100


class Brake(BaseRoutine):
    @staticmethod
    def run(agent: VirxERLU):
        if brake_handler(agent):
            agent.pop()
brake = Brake  # legacy


class Flip(BaseRoutine):
    # Flip takes a vector in local coordinates and flips/dodges in that direction
    # cancel causes the flip to cancel halfway through, which can be used to half-flip
    def __init__(self, vector: Vector, cancel: bool=False, face: Optional[Vector]=None):
        target_angle = math.atan2(vector.y, vector.x)
        self.yaw = math.sin(target_angle)
        self.pitch = -math.cos(target_angle)
        self.throttle = -1 if self.pitch > 0 else 1

        self.cancel = cancel
        self.face = face
        # the time the jump began
        self.time = -1
        # keeps track of the frames the jump button has been released
        self.counter = 0

    def run(self, agent: VirxERLU, manual: bool=False):
        if self.time == -1:
            self.time = agent.time

        elapsed = agent.time - self.time
        agent.controller.throttle = self.throttle

        if elapsed < 0.1:
            agent.controller.jump = True
        elif elapsed >= 0.1 and self.counter < 3:
            agent.controller.pitch = self.pitch
            agent.controller.yaw = self.yaw
            agent.controller.jump = False
            self.counter += 1
        elif agent.me.airborne and (elapsed < 0.4 or (not self.cancel and elapsed < 0.9)):
            agent.controller.pitch = self.pitch
            agent.controller.yaw = self.yaw
            agent.controller.jump = True
        else:
            if not manual:
                agent.pop()
            agent.push(Recovery(self.face))
            if manual:
                return True
flip = Flip  # legacy


class GoTo:
    # Drives towards a designated (stationary) target
    # Optional vector controls where the car should be pointing upon reaching the target
    # Brake brings the car to slow down to 0 when it gets to it's destination
    # Slow is for small targets, and it forces the car to slow down a bit when it gets close to the target
    def __init__(self, target: Vector, vector: Optional[Vector]=None, brake: bool=False, slow: bool=False):
        self.target = target
        self.vector = vector
        self.brake = brake
        self.slow = slow

        self.f_brake = False
        self.rule1_timer = -1

    def run(self, agent: VirxERLU, manual: bool=False):
        car_to_target = self.target - agent.me.location
        distance_remaining = car_to_target.flatten().magnitude()

        agent.dbg_2d(f"Distance to target: {round(distance_remaining)}")
        agent.line(self.target - Vector(z=500), self.target + Vector(z=500), (255, 0, 255))

        if self.brake and (self.f_brake or distance_remaining * 0.95 < (agent.me.local_velocity().x ** 2 * -1) / (2 * -utils.BRAKE_ACC)):
            self.f_brake = True
            done = brake_handler(agent)
            if done and not manual:
                agent.pop()
            return

        if not self.brake and not manual and distance_remaining < 320:
            agent.pop()
            return

        final_target = self.target.copy().flatten()

        if self.vector is not None:
            # See comments for adjustment in jump_shot for explanation
            side_of_vector = utils.sign(self.vector.cross(Vector(z=1)).dot(car_to_target))
            car_to_target_perp = car_to_target.cross(Vector(z=side_of_vector)).normalize()
            adjustment = car_to_target.angle2D(self.vector) * distance_remaining / 3.14
            final_target += car_to_target_perp * adjustment

        final_target = utils.cap_in_field(agent, final_target)  # Some adjustment to the final target to ensure it's inside the field and we don't try to drive through any goalposts to reach it
        local_target = agent.me.local_location(final_target)
        angle_to_target = abs(Vector(x=1).angle2D(local_target))
        true_angle_to_target = abs(Vector(x=1).angle2D(agent.me.local_location(self.target)))
        direction = 1 if angle_to_target < 1.6 or agent.me.local_velocity().x > 1000 else -1
        agent.dbg_2d(f"Angle to target: {round(angle_to_target, 1)}")

        travel_speed = 2300 if (distance_remaining > 1280 or not self.slow) else utils.cap(distance_remaining * 2, 1200, 2300)
        velocity = utils.defaultDrive(agent, travel_speed * direction, local_target)[1]
        if distance_remaining < 2560: agent.controller.boost = False

        # this is to break rule 1's with TM8'S ONLY
        # 251 is the distance between center of the 2 longest cars in the game, with a bit extra
        if len(agent.friends) > 0 and agent.me.local_velocity().x < 50 and agent.controller.throttle == 1 and min(agent.me.location.flat_dist(car.location) for car in agent.friends) < 251:
            if self.rule1_timer == -1:
                self.rule1_timer = agent.time
            elif agent.time - self.rule1_timer > 1.5:
                agent.push(Flip(Vector(y=250)))
                return
        elif self.rule1_timer != -1:
            self.rule1_timer = -1

        dodge_time = distance_remaining / (abs(velocity) + utils.dodge_impulse(agent)) - 0.8

        if agent.me.airborne:
            agent.push(Recovery(self.target))
        elif dodge_time >= 1.2 and agent.time - agent.me.land_time > 0.5:
            if agent.me.boost < 48 and angle_to_target < 0.03 and (true_angle_to_target < 0.1 or distance_remaining > 4480) and velocity > 600:
                agent.push(Flip(agent.me.local_location(self.target)))
            elif direction == -1 and velocity < 200:
                agent.push(Flip(agent.me.local_location(self.target), True))
goto = GoTo # legacy


class GoToBoost(BaseRoutine):
    # very similar to goto() but designed for grabbing boost
    def __init__(self, boost: boost_object):
        self.boost = boost
        self.goto = GoTo(self.boost.location, slow=not self.boost.large)

    def run(self, agent: VirxERLU):
        if not self.boost.active or agent.me.boost == 100:
            agent.pop()
            return

        self.goto.vector = agent.ball.location if not self.boost.large and self.boost.location.flat_dist(agent.me.location) > 640 else None
        self.goto.run(agent, manual=True)
goto_boost = GoToBoost  # legacy


class GenericKickoff(BaseRoutine):
    def __init__(self):
        self.start_time = -1
        self.flip = False

    def run(self, agent: VirxERLU):
        if self.start_time == -1:
            self.start_time = agent.time

        if self.flip or agent.time - self.start_time > 3:
            agent.kickoff_done = True
            agent.pop()
            return

        target = agent.ball.location + Vector(y=(200 if agent.gravity.z < -600 and agent.gravity.z > -700 else 50) * utils.side(agent.team))
        local_target = agent.me.local_location(target)

        utils.defaultPD(agent, local_target)
        agent.controller.throttle = 1
        agent.controller.boost = True

        distance = local_target.magnitude()

        if distance < 550:
            self.flip = True
            agent.push(Flip(agent.me.local_location(agent.foe_goal.location)))
generic_kickoff = GenericKickoff()  # legacy

