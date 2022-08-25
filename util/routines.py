from __future__ import annotations

import math
from traceback import print_exc
from typing import Optional

import numpy as np
import virx_erlu_rlib as rlru
from numba import njit

from util import utils
from util.agent import BaseRoutine, Vector, VirxERLU, BoostPad

MAX_JUMP_HOLD_TIME = 0.2

class GroundShot(BaseRoutine):
    def __init__(self, intercept_time: float, target_id: int):
        self.intercept_time = intercept_time
        self.target_id = target_id
        self.end_next_tick = False
        self.distance = []

    def update(self, shot: GroundShot):
        if self.intercept_time < shot.intercept_time + 0.1:
            return

        try:
            rlru.confirm_target(shot.target_id)
        except Exception:
            print_exc()
            return

        rlru.remove_target(self.target_id)

        self.intercept_time = shot.intercept_time
        self.target_id = shot.target_id
        self.end_next_tick = False

    def run(self, agent: VirxERLU):
        if self.end_next_tick:
            agent.pop()
            return

        if agent.me.airborne:
            agent.push(Recovery())
            return

        future_ball_location = Vector(*rlru.get_slice(self.intercept_time).location)

        agent.sphere(future_ball_location, agent.ball_radius, agent.renderer.purple())

        T = self.intercept_time - agent.time

        agent.dbg_3d(f"Time to intercept: {round(T, 1)}")

        try:
            shot_info = rlru.get_data_for_shot_with_target(self.target_id)
        except IndexError as e:
            # Either the target has been removed, never existed, or something else has gone wrong
            agent.print(f"WARNING: {e}")
            agent.pop()
            return
        except AssertionError as e:
            # One of our predictions was incorrect
            # We could've gotten bumped, the ball bounced weird, or something else
            agent.print(f"WARNING: {e}")
            agent.pop()
            return
        except ValueError:
            agent.print(f"WARNING: We ran out of time to execute the shot")
            agent.pop()
            return

        self.distance.append(Vector(*shot_info.current_path_point).flat_dist(agent.me.location))

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

        flip_time = utils._fcap(320. / speed_required - 0.05, 0.1, 0.25)
        if flip_time - 0.05 < T < flip_time:
            flip_dir = agent.me.local_location(future_ball_location + Vector(shot_info.shot_vector[0], shot_info.shot_vector[1]) * agent.ball_radius)
            self.end_next_tick = True
            agent.push(Flip(flip_dir))

    def on_push(self):
        rlru.confirm_target(self.target_id)

    def pre_pop(self):
        if len(self.distance) > 0:
            print(sum(self.distance) / len(self.distance))
        rlru.remove_target(self.target_id)
ground_shot = GroundShot  # legacy


class JumpShot(BaseRoutine):
    def __init__(self, intercept_time: float, target_id: int):
        self.intercept_time = intercept_time
        self.target_id = target_id
        self.jumping = False
        self.jump_time = -1
        self.last_jump = None
        self.dodge_params = None
        self.recovering = False
        self.distance = []

    def update(self, shot: JumpShot):
        if self.intercept_time < shot.intercept_time + 0.1 or self.jumping:
            return

        try:
            rlru.confirm_target(shot.target_id)
        except Exception:
            print_exc()
            return

        rlru.remove_target(self.target_id)

        self.intercept_time = shot.intercept_time
        self.target_id = shot.target_id

    def run(self, agent: VirxERLU):
        T = self.intercept_time - agent.time
        agent.dbg_3d(f"Time to intercept: {round(T, 1)}")

        if self.recovering:
            if T >= -0.25:
                agent.controller.yaw = self.dodge_params[0]
                agent.controller.pitch = self.dodge_params[1]
            else:
                agent.pop()
                agent.push(Recovery())
            return

        if agent.me.airborne and not self.jumping:
            agent.push(Recovery())
            return

        future_ball_location = Vector(*rlru.get_slice(self.intercept_time).location)
        agent.sphere(future_ball_location, agent.ball_radius, agent.renderer.purple())

        if T <= 0.1 and self.jumping and self.dodge_params is not None and agent.ball.last_touch.car.index == agent.me.index and abs(self.intercept_time - agent.ball.last_touch.time) < 0.1:
            self.recovering = True
            return

        try:
            shot_info = rlru.get_data_for_shot_with_target(self.target_id)
        except IndexError as e:
            # Either the target has been removed, never existed, or something else has gone wrong
            agent.print(f"WARNING: {e}")
            agent.pop()
            return
        except AssertionError as e:
            # One of our predictions was incorrect
            # We could've gotten bumped, the ball bounced weird, or something else
            agent.print(f"WARNING: {e}")
            agent.pop()
            return
        except ValueError:
            agent.print(f"WARNING: We ran out of time to execute the shot")
            agent.pop()
            return

        current_path_point = Vector(*shot_info.current_path_point)
        self.distance.append(current_path_point.flat_dist(agent.me.location))

        final_target = Vector(*shot_info.final_target)
        agent.point(final_target, agent.renderer.red())

        shot_vector = Vector(*shot_info.shot_vector)
        agent.line(future_ball_location + shot_vector * agent.ball_radius, future_ball_location + shot_vector * agent.ball_radius * 3, agent.renderer.lime())

        if len(shot_info.path_samples) > 2:
            agent.polyline(tuple(Vector(sample[0], sample[1], 30) for sample in shot_info.path_samples), agent.renderer.lime())
        else:
            agent.line(agent.me.location, final_target, agent.renderer.lime())

        distance_remaining = shot_info.distance_remaining
        speed_required = min(distance_remaining / T, 2300)
        local_final_target = agent.me.local_location(Vector(final_target.x, final_target.y, agent.me.location.z))

        agent.dbg_2d(f"Required jump time: {round(shot_info.required_jump_time, 1)}")
        if T < shot_info.required_jump_time:
            self.jumping = True

        if not self.jumping:
            distance = agent.me.right.dot(current_path_point - agent.me.location)
            # print(distance, flush=True)
            utils.defaultDrive(agent, speed_required, local_final_target, distance=distance)
            return

        if not agent.me.airborne:
            self.jump_time = agent.time

        if agent.time - self.jump_time < MAX_JUMP_HOLD_TIME:
            agent.controller.jump = True
            utils.defaultPD(agent, local_final_target)

        if T < 0.1:
            if self.dodge_params is None:
                flip_dir = agent.me.local_location(future_ball_location + shot_vector * agent.ball_radius)
                target_angle = math.atan2(flip_dir.y, flip_dir.x)
                yaw = math.sin(target_angle)
                neg_pitch = math.cos(target_angle)  # negative pitch is down which flips us forward, so cosine and pitch are inversly related
                self.dodge_params = (
                    yaw,
                    -neg_pitch,
                    utils._fsign(neg_pitch),  # throttle and cosine are directly related
                )

            agent.controller.yaw = self.dodge_params[0]
            agent.controller.pitch = self.dodge_params[1]
            agent.controller.throttle = self.dodge_params[2]

            if self.last_jump is None:
                self.last_jump = agent.time

            if agent.time - self.last_jump < 0.05:
                agent.controller.jump = False
                self.last_jump = False
            else:
                agent.controller.jump = True
        elif agent.me.airborne:
            utils.defaultThrottle(agent, speed_required)

    def on_push(self):
        rlru.confirm_target(self.target_id)

    def pre_pop(self):
        if len(self.distance) > 0:
            print(sum(self.distance) / len(self.distance))
        rlru.remove_target(self.target_id)
jump_shot = JumpShot  # legacy


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
    agent.controller.throttle = -utils.cap(speed / (utils.BRAKE_ACC * agent.delta_time), -1, 1)
    # threshold of "we're close enough to stopping"
    return abs(speed) < 100


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
            side_of_vector = utils._fsign(self.vector.cross(Vector(z=1)).dot(car_to_target))
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
    def __init__(self, boost: BoostPad):
        self.boost = boost
        self.goto = GoTo(self.boost.location, slow=not self.boost.large)

    def run(self, agent: VirxERLU):
        if not self.boost.active or agent.me.boost == 100:
            agent.pop()
            return

        self.goto.vector = agent.ball.location if not self.boost.large and self.boost.location.flat_dist(agent.me.location) > 640 else None
        self.goto.run(agent, manual=True)
goto_boost = GoToBoost  # legacy


class FaceTarget(BaseRoutine):
    def __init__(self, target: Optional[Vector]=None, ball: bool=False):
        self.target = target
        self.ball = ball
        self.start_loc = None
        self.counter = 0

    @staticmethod
    def get_ball_target(agent: VirxERLU) -> Vector:
        ball = agent.ball.location
        return Vector(ball.x, ball.y)

    def run(self, agent: VirxERLU):
        if self.ball:
            target = self.get_ball_target(agent) - agent.me.location
        else:
            target = agent.me.velocity if self.target is None else self.target - agent.me.location

        if agent.gravity.z < -550 and agent.gravity.z > -750:
            if self.counter == 0 and abs(Vector(x=1).angle(target)) <= 0.05:
                agent.pop()
                return

            if self.counter == 0 and agent.me.airborne:
                self.counter = 3

            if self.counter < 3:
                self.counter += 1

            target = agent.me.local(target.flatten())
            if self.counter < 3:
                agent.controller.jump = True
            elif agent.me.airborne and abs(Vector(x=1).angle(target)) > 0.05:
                utils.defaultPD(agent, target)
            else:
                agent.pop()
        else:
            target = agent.me.local(target.flatten())
            angle_to_target = abs(Vector(x=1).angle(target))
            if angle_to_target > 0.1:
                if self.start_loc is None:
                    self.start_loc = agent.me.location

                direction = -1 if angle_to_target < 1.57 else 1

                agent.controller.steer = utils.cap(target.y / 100, -1, 1) * direction
                agent.controller.throttle = direction
                agent.controller.handbrake = True
            else:
                agent.pop()
                if self.start_loc is not None:
                    agent.push(GoTo(self.start_loc, target, True))
face_target = FaceTarget  # legacy


class Retreat(BaseRoutine):
    def __init__(self):
        self.goto = GoTo(Vector(), brake=True)

    def run(self, agent: VirxERLU):
        ball = self.get_ball_loc(agent, render=True)
        target = self.get_target(agent, ball=ball)

        if Shadow.is_viable(agent, ignore_distance=True):
            agent.pop()
            agent.push(Shadow())
            return

        self.goto.target = target
        self.goto.run(agent)

    def is_viable(self, agent: VirxERLU) -> bool:
        return agent.me.location.flat_dist(self.get_target(agent)) > 320 and not Shadow.is_viable(agent, ignore_distance=True)

    @staticmethod
    def get_ball_loc(agent: VirxERLU, render: bool=False) -> Vector:
        ball_slice = agent.ball.location

        ball = Vector(ball_slice.x, ball_slice.y)
        if render: agent.sphere(ball + Vector(z=agent.ball_radius), agent.ball_radius, color=agent.renderer.black())
        ball.y *= utils.side(agent.team)

        if ball.y < agent.ball.location.y * utils.side(agent.team):
            ball = Vector(agent.ball.location.x, agent.ball.location.y * utils.side(agent.team) + 640)

        return ball

    @staticmethod
    def get_target(agent: VirxERLU, ball: Optional[Vector]=None) -> Vector:
        if ball is None:
            ball = Retreat.get_ball_loc(agent)

        friends = np.array(tuple(friend.location._np for friend in agent.friends), dtype=np.float32, ndmin=2)

        friend_goal = np.array((
            agent.friend_goal.location._np,
            agent.friend_goal.left_post._np,
            agent.friend_goal.right_post._np
        ), dtype=np.float32)

        return Vector(np_arr=Retreat._get_target(friends, friend_goal, ball._np, np.int32(agent.team)))

    @staticmethod
    @njit('Array(float32, 1, "C")(Array(float32, 2, "C"), Array(float32, 2, "C"), Array(float32, 1, "C"), int32)', fastmath=True, cache=True)
    def _get_target(friends: np.ndarray, friend_goal: np.ndarray, ball: np.ndarray, team: int) -> np.ndarray:
        target = None
        
        self_team = utils.side(team)

        horizontal_offset = 150
        outside_goal_offset = -125
        inside_goal_offset = 150

        if ball[1] < -640:
            target = friend_goal[0].copy()
        elif ball[0] * self_team < friend_goal[2][0] * self_team:
            target = friend_goal[2].copy()

            while utils.friend_near_target(friends, target):
                target[0] = (target[0] * self_team + horizontal_offset * self_team) * self_team
        elif ball[0] * self_team > friend_goal[1][0] * self_team:
            target = friend_goal[1].copy()

            while utils.friend_near_target(friends, target):
                target[0] = (target[0] * self_team - horizontal_offset * self_team) * self_team
        else:
            target = friend_goal[0].copy()
            target[0] = ball[0]

            while utils.friend_near_target(friends, target):
                target[0] = (target[0] * self_team - horizontal_offset * utils._fsign(ball[0]) * self_team) * self_team

        target[1] += (inside_goal_offset if abs(target[0]) < 800 else outside_goal_offset) * self_team
        target[2] = 0

        return target
retreat = Retreat # legacy


class Shadow(BaseRoutine):
    def __init__(self):
        self.goto = GoTo(Vector(), brake=True)

    def run(self, agent: VirxERLU):
        ball_loc = self.get_ball_loc(agent, True)
        target = self.get_target(agent, ball_loc)

        if self.switch_to_retreat(agent, ball_loc, target):
            agent.pop()
            agent.push(Retreat())
            return

        self_to_target = agent.me.location.flat_dist(target)

        if self_to_target < 100 * (agent.me.velocity.magnitude() / 500) and ball_loc.y < -640 and agent.me.velocity.magnitude() < 50 and abs(Vector(x=1).angle2D(agent.me.local_location(agent.ball.location))) > 1:
            agent.pop()
            if len(agent.friends) > 1:
                agent.push(FaceTarget(ball=True))
        else:
            self.goto.target = target
            self.goto.vector = ball_loc * Vector(y=utils.side(agent.team)) if target.y * utils.side(agent.team) < 1280 else None
            self.goto.run(agent)

    @staticmethod
    def switch_to_retreat(agent: VirxERLU, ball: Vector, target: Vector) -> bool:
        return agent.me.location.y * utils.side(agent.team) < ball.y or ball.y > 2560 or target.y * utils.side(agent.team) > 4480

    @staticmethod
    def is_viable(agent: VirxERLU, ignore_distance: bool=False, ignore_retreat=False) -> bool:
        ball_loc = Shadow.get_ball_loc(agent)
        target = Shadow.get_target(agent, ball_loc)

        return (ignore_distance or agent.me.location.flat_dist(target) > 320) and (ignore_retreat or not Shadow.switch_to_retreat(agent, ball_loc, target))

    @staticmethod
    def get_ball_loc(agent: VirxERLU, render: bool=False) -> Vector:
        ball_slice = agent.ball.location
        ball_loc = Vector(ball_slice.x, ball_slice.y)
        if render: agent.sphere(ball_loc + Vector(z=agent.ball_radius), agent.ball_radius, color=agent.renderer.black())
        ball_loc.y *= utils.side(agent.team)

        if ball_loc.y < -2560 or (ball_loc.y < agent.ball.location.y * utils.side(agent.team)):
            ball_loc = Vector(agent.ball.location.x, agent.ball.location.y * utils.side(agent.team) - 640)

        return ball_loc

    @staticmethod
    def get_target(agent: VirxERLU, ball_loc: Optional[Vector]=None) -> Vector:
        if ball_loc is None:
            ball_loc = Shadow.get_ball_loc(agent)

        if len(agent.friends) > 0:
            distance = 3840
        else:
            distances = [
                1920,
                3840,
                4800
            ]
            distance = distances[min(len(agent.friends), 2)]

        target = Vector(y=(ball_loc.y + distance) * utils.side(agent.team))
        if target.y * utils.side(agent.team) > -1280:
            # find the proper x coord for us to stop a shot going to the net
            # y = mx + b <- yes, finally! 7th grade math is paying off xD
            p1 = Retreat.get_target(agent)
            p2 = ball_loc * Vector(x=1, y=utils.side(agent.team))
            try:
                m = (p2.y - p1.y) / (p2.x - p1.x)
                b = p1.y - (m * p1.x)
                # x = (y - b) / m
                target.x = (target.y - b) / m
            except ZeroDivisionError:
                target.x = 0
        else:
            target.x = (abs(ball_loc.x) + 640) * utils._fsign(ball_loc.x)

        return Vector(target.x, target.y)
shadow = Shadow # legacy


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

