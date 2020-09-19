import virxrlcu
from util.utils import (Vector, backsolve, cap, defaultPD, defaultThrottle,
                        shot_valid, side, sign)

max_speed = 2300
throttle_accel = 100 * (2/3)
brake_accel = Vector(x=-3500)
boost_per_second = 33 + (1/3)
min_boost_time = 1/30
jump_max_duration = 0.2
jump_speed = 291 + (2/3)
jump_acc = 1458 + (1/3)


class atba:
    def __init__(self, exit_distance=500, exit_flip=True):
        self.exit_distance = exit_distance
        self.exit_flip = exit_flip

    def run(self, agent):
        target = agent.ball.location
        car_to_target = target - agent.me.location
        distance_remaining = car_to_target.flatten().magnitude()

        angle_to_target = abs(Vector(x=1).angle2D(agent.me.local_location(target)))
        direction = 1 if angle_to_target < 2.2 else -1
        local_target = agent.me.local_location(target)

        # Some adjustment to the final target to ensure it's inside the field and we don't try to drive through any goalposts to reach it
        if abs(agent.me.location.y) > 5120 - (agent.me.hitbox.width / 2):
            local_target.x = cap(local_target.x, -750, 750)

        angles = defaultPD(agent, local_target)
        defaultThrottle(agent, 1400)

        agent.controller.handbrake = angle_to_target > 1.54

        velocity = agent.me.local_velocity().x
        if distance_remaining < self.exit_distance:
            agent.pop()
            if self.exit_flip:
                agent.push(flip(local_target))
        elif angle_to_target < 0.05 and velocity > 600 and velocity < 1400 and distance_remaining / velocity > 3:
            agent.push(flip(local_target))
        elif angle_to_target >= 2 and distance_remaining > 1000 and velocity < 200 and distance_remaining / abs(velocity) > 2:
            agent.push(flip(local_target, True))
        elif agent.me.airborne:
            agent.push(recovery(local_target))


class wave_dash:
    def __init__(self, target=None):
        self.step = -1
        # 0 = forward, 1 = right, 2 = backwards, 3 = left
        self.direction = 0
        self.recovery = recovery()

        if target is not None:
            largest_direction = max(abs(target.x), abs(target.y))

            dir_switch = {
                abs(target.x): 0,
                abs(target.y): 1
            }

            self.direction = dir_switch[largest_direction]

            if (self.direction == 0 and target.x < 0) or (self.direction and target.y < 0):
                self.direction += 2

    def run(self, agent):
        self.step += 1

        target_switch = {
            0: agent.me.forward.flatten()*100 + Vector(z=50),
            1: agent.me.forward.flatten()*100,
            2: agent.me.forward.flatten()*100 - Vector(z=50),
            3: agent.me.forward.flatten()*100
        }

        target_up = {
            0: agent.me.local(Vector(z=1)),
            1: agent.me.local(Vector(y=-50, z=1)),
            2: agent.me.local(Vector(z=1)),
            3: agent.me.local(Vector(y=50, z=1))
        }

        defaultPD(agent, agent.me.local(target_switch[self.direction]), up=target_up[self.direction])
        if self.direction == 0:
            agent.controller.throttle = 1
        elif self.direction == 2:
            agent.controller.throttle = -1
        else:
            agent.controller.handbrake = True

        if self.step < 1:
            agent.controller.jump = True
        elif self.step < 4:
            pass
        elif not agent.me.airborne:
            agent.pop()
        elif agent.me.location.z + (agent.me.velocity.z * 0.2) < 5:
            agent.controller.jump = True
            agent.controller.yaw = 0
            if self.direction in {0, 2}:
                agent.controller.roll = 0
                agent.controller.pitch = -1 if self.direction is 0 else 1
            else:
                agent.controller.roll = -1 if self.direction is 1 else 1
                agent.controller.pitch = 0


class double_jump:
    def __init__(self, intercept_time, shot_vector):
        self.intercept_time = intercept_time
        self.shot_vector = shot_vector
        self.ball_location = None
        # braking
        self.brake = brake()
        # Flags for what part of the routine we are in
        self.jumping = False
        self.dodged = False
        self.jump_time = -1
        self.counter = 0

    def update(self, shot):
        self.intercept_time = shot.intercept_time
        self.shot_vector = shot.shot_vector

    def run(self, agent):
        # This routine is the same as jump_shot, but it's designed to hit the ball above 250uus (and below 551uus or so) without any boost
        if not agent.shooting:
            agent.shooting = True

        T = self.intercept_time - agent.time
        # Capping T above 0 to prevent division problems
        time_remaining = cap(T, 0.000001, 6)

        slice_n = round(T * 60)
        agent.dbg_2d(f"Shot slice #: {slice_n}")

        if T > 0.3 or self.ball_location is None:
            ball = agent.ball_prediction_struct.slices[slice_n].physics.location

            self.ball_location = Vector(ball.x, ball.y, ball.z)
            self.dodge_point = self.ball_location - (self.shot_vector * agent.best_shot_value)

            if self.dodge_point.z > 490 or self.dodge_point.z < 380:
                agent.pop()
                return

        car_to_ball = self.ball_location - agent.me.location
        # whether we should go forwards or backwards
        angle_to_target = abs(Vector(x=1).angle2D(agent.me.local_location(self.dodge_point)))
        # whether we are to the left or right of the shot vector
        side_of_shot = sign(self.shot_vector.cross(Vector(z=1)).dot(car_to_ball))

        final_target = self.dodge_point.copy()

        # Some adjustment to the final target to ensure it's inside the field and we don't try to drive through any goalposts to reach it
        if abs(agent.me.location.y) > 5120 - (agent.me.hitbox.length / 2):
            final_target.x = cap(final_target.x, -750, 750)

        car_to_dodge_point = final_target - agent.me.location
        car_to_dodge_perp = car_to_dodge_point.cross(Vector(z=side_of_shot))  # perpendicular
        distance_remaining = car_to_dodge_point.flatten().magnitude()

        acceleration_required = backsolve(self.dodge_point, agent.me, time_remaining, Vector() if not self.jumping else agent.gravity)
        local_acceleration_required = agent.me.local_velocity(acceleration_required)
        # The adjustment causes the car to circle around the dodge point in an effort to line up with the shot vector
        # The adjustment slowly decreases to 0 as the bot nears the time to jump
        adjustment = car_to_dodge_point.angle2D(self.shot_vector) * distance_remaining / 2  # size of adjustment
        # controls how soon car will jump based on acceleration required
        # we set this based on the time remaining
        # bigger = later, which allows more time to align with shot vector
        # smaller = sooner
        jump_threshold = cap(T * 200, 250, 584)
        # factoring in how close to jump we are
        adjustment *= (cap(jump_threshold - (acceleration_required.z), 0, jump_threshold) / jump_threshold)
        # we don't adjust the final target if we are already jumping
        final_target += ((car_to_dodge_perp.normalize() * adjustment) if not self.jumping else 0) + Vector(z=50)

        distance_remaining = (final_target - agent.me.location).flatten().magnitude()
        local_final_target = agent.me.local_location(final_target)

        # drawing debug lines to show the dodge point and final target (which differs due to the adjustment)
        agent.line(agent.me.location, self.dodge_point, agent.renderer.white())
        agent.line(self.dodge_point-Vector(z=100), self.dodge_point+Vector(z=100), agent.renderer.green())

        vf = agent.me.velocity + agent.gravity * T

        distance = agent.me.local_location(self.dodge_point).x if agent.me.airborne else distance_remaining
        speed_required = 2300 if distance_remaining > 2560 else distance / time_remaining
        agent.dbg_2d(f"Speed required: {speed_required}")

        if not self.jumping:
            velocity = agent.me.local_velocity().x

            local_dodge_point = agent.me.local_location(self.dodge_point.flatten())
            local_vf = agent.me.local(vf.flatten())

            if T <= 0 or not virxrlcu.double_jump_shot_is_viable(T + 0.3, agent.boost_accel, agent.me.location.dist(self.ball_location), self.shot_vector.tuple(), agent.me.forward.tuple(), agent.me.boost if agent.boost_amount != 'unlimited' else 100000, velocity):
                # If we're out of time or the ball was hit away or we just can't get enough speed, pop
                agent.pop()
                agent.shooting = False
                agent.shot_weight = -1
                agent.shot_time = -1
                if agent.me.airborne:
                    agent.push(ball_recovery())
            elif abs(local_dodge_point.y) < 92 and local_vf.x >= local_dodge_point.x:
                # Switch into the jump when the upward acceleration required reaches our threshold, and our lateral acceleration is negligible, and we're close enough to the point in time where the ball will be at the given location
                self.jumping = True
            elif agent.boost_amount != 'unlimited' and angle_to_target < 0.03 and velocity > 600 and velocity < speed_required - 500 and distance_remaining / velocity > 3:
                if agent.gravity.z < -450:
                    agent.push(wave_dash())
                else:
                    agent.push(flip(local_final_target))
            elif agent.boost_amount != 'unlimited' and angle_to_target >= 2 and distance_remaining > 1000 and velocity < 200:
                agent.push(flip(local_final_target, True))
            elif agent.me.airborne:
                agent.push(recovery(local_final_target))
            else:
                defaultPD(agent, local_final_target)
                defaultThrottle(agent, speed_required)
                agent.controller.handbrake = angle_to_target > 1.54
        else:
            # Mark the time we started jumping so we know when to dodge
            if self.jump_time == -1:
                self.jump_time = agent.time

            jump_elapsed = agent.time - self.jump_time

            tau = jump_max_duration - jump_elapsed
            xf = agent.me.location + agent.me.velocity * T + 0.5 * agent.gravity * T * T

            if jump_elapsed == 0:
                vf += agent.me.up * jump_speed
                xf += agent.me.up * jump_speed * T

            vf += agent.me.up * jump_acc * tau
            xf += agent.me.up * jump_acc * tau * (T - 0.5 * tau)

            vf += agent.me.up * jump_speed
            xf += agent.me.up * jump_speed * (T - tau)

            delta_x = self.dodge_point - xf
            direction = delta_x.normalize()

            if abs(agent.me.forward.dot(direction)) > 0.5:
                delta_v = delta_x.dot(agent.me.forward) / T
                if agent.me.boost > 0 and delta_v >= agent.boost_accel * min_boost_time:
                    agent.controller.boost = True
                else:
                    agent.controller.throttle = cap(delta_v / (throttle_accel * min_boost_time), -1, 1)

            if T <= -0.4 or (not agent.me.airborne and self.counter > 0):
                agent.pop()
                agent.shooting = False
                agent.shot_weight = -1
                agent.shot_time = -1
                agent.push(ball_recovery())
            elif jump_elapsed < jump_max_duration and vf.z <= self.dodge_point.z:
                agent.controller.jump = True
            elif self.counter < 4:
                self.counter += 1

            if self.counter == 3:
                agent.controller.jump = True
            elif self.counter == 4:
                defaultPD(agent, agent.me.local_location(self.dodge_point), upside_down=True)

            if self.counter < 3:
                defaultPD(agent, agent.me.local((self.dodge_point - agent.me.location).flatten()))

        l_vf = vf + agent.me.location
        agent.line(l_vf-Vector(z=100), l_vf+Vector(z=100), agent.renderer.red())


class Aerial:
    def __init__(self, intercept_time, shot_vector, fast_aerial=True):
        self.intercept_time = intercept_time
        self.shot_vector = shot_vector
        self.fast_aerial = fast_aerial
        self.target = None

        self.jumping = False
        self.dodging = False
        self.ceiling = False
        self.time = -1
        self.jump_time = -1
        self.counter = 0

    def update(self, shot):
        self.intercept_time = shot.intercept_time
        self.shot_vector = shot.shot_vector
        self.fast_aerial = shot.fast_aerial

    def run(self, agent):
        if not agent.shooting:
            agent.shooting = True

        if self.time == -1:
            self.time = agent.time

        elapsed = agent.time - self.time
        T = self.intercept_time - agent.time
        xf = agent.me.location + agent.me.velocity * T + 0.5 * agent.gravity * T * T
        vf = agent.me.velocity + agent.gravity * T

        slice_n = math.ceil(T * 60)
        agent.dbg_2d(f"Shot slice #: {slice_n}")

        if T > 0.1 or self.target is None:
            ball = agent.ball_prediction_struct.slices[slice_n].physics.location

            self.ball = Vector(ball.x, ball.y, ball.z)
            self.target = self.ball - (self.shot_vector * agent.best_shot_value * 0.8)

            if agent.me.location.z > 2044 - agent.me.hitbox.height * 1.1:
                self.ceiling = True
                self.target -= Vector(z=92)

        if not self.ceiling and (self.jumping or not agent.me.airborne):
            agent.dbg_2d("Jumping")

            if not self.jumping or not agent.me.airborne:
                self.jumping = True
                self.jump_time = agent.time
                self.counter = 0

            jump_elapsed = agent.time - self.jump_time

            tau = jump_max_duration - jump_elapsed

            if jump_elapsed == 0:
                vf += agent.me.up * jump_speed
                xf += agent.me.up * jump_speed * T

            vf += agent.me.up * jump_acc * tau
            xf += agent.me.up * jump_acc * tau * (T - 0.5 * tau)

            if self.fast_aerial:
                vf += agent.me.up * jump_speed
                xf += agent.me.up * jump_speed * (T - tau)

                if jump_elapsed < jump_max_duration:
                    agent.controller.jump = True
                elif self.counter < 6:
                    self.counter += 1

                if self.counter == 3:
                    agent.controller.jump = True
                    self.dodging = True
                elif self.counter == 6:
                    self.dodging = self.jumping = False
            elif jump_elapsed < jump_max_duration:
                agent.controller.jump = True
            else:
                self.jumping = False

        if self.ceiling:
            agent.dbg_2d(f"Ceiling shot")

        delta_x = self.target - xf
        direction = delta_x.normalize()

        agent.line(agent.me.location, self.target, agent.renderer.white())
        c_vf = vf + agent.me.location
        agent.line(c_vf - Vector(z=100), c_vf + Vector(z=100), agent.renderer.blue())
        agent.line(xf - Vector(z=100), xf + Vector(z=100), agent.renderer.red())
        agent.line(self.target - Vector(z=100), self.target + Vector(z=100), agent.renderer.green())

        if not self.dodging:
            target = delta_x if delta_x.magnitude() > 50 else (self.target - agent.me.location)

            if self.jumping:
                target = target.flatten()

            target = agent.me.local(target)
            if abs(Vector(x=1).angle(target)) > 0.005:
                defaultPD(agent, target, upside_down=self.shot_vector.z < 0 and not self.jumping)

        if abs(agent.me.forward.dot(direction)) > 0.5:
            delta_v = delta_x.dot(agent.me.forward) / T
            if agent.me.boost > 0 and delta_v >= agent.boost_accel * min_boost_time:
                agent.controller.boost = True
            else:
                agent.controller.throttle = cap(delta_v / (throttle_accel * min_boost_time), -1, 1)

        if T <= 0 or (not self.jumping and not agent.me.airborne) or (not self.jumping and T > 2 and self.fast_aerial and not virxrlcu.aerial_shot_is_viable(T + 0.3, 144, agent.boost_accel, agent.gravity.tuple(), agent.me.location.tuple(), agent.me.velocity.tuple(), agent.me.up.tuple(), agent.me.forward.tuple(), 1 if agent.me.airborne else -1, agent.me.boost if agent.boost_amount != 'unlimited' else 100000, self.ball.tuple())):
            agent.pop()
            agent.shooting = False
            agent.shot_weight = -1
            agent.shot_time = -1
            agent.push(ball_recovery())
        elif (self.ceiling and self.target.dist(agent.me.location) < 92 + agent.me.hitbox.length and not agent.me.doublejumped and agent.me.location.z < agent.ball.location.z + 92 and self.target.y * side(agent.team) > -4240) or (not self.ceiling and not self.fast_aerial and self.target.dist(agent.me.location) < 92 + agent.me.hitbox.length and not agent.me.doublejumped):
            agent.dbg_2d("Flipping")
            agent.controller.jump = True
            local_target = agent.me.local_location(self.target)
            agent.controller.pitch = abs(local_target.x) * -sign(local_target.x)
            agent.controller.yaw = abs(local_target.y) * sign(local_target.y)


class flip:
    # Flip takes a vector in local coordinates and flips/dodges in that direction
    # cancel causes the flip to cancel halfway through, which can be used to half-flip
    def __init__(self, vector, cancel=False):
        self.vector = vector.normalize()
        self.pitch = abs(self.vector.x) * -sign(self.vector.x)
        self.yaw = abs(self.vector.y) * sign(self.vector.y)
        self.cancel = cancel
        # the time the jump began
        self.time = -1
        # keeps track of the frames the jump button has been released
        self.counter = 0

    def run(self, agent, manual=False):
        if agent.gravity.z >= 3250:
            agent.pop()

        if self.time == -1:
            elapsed = 0
            self.time = agent.time
        else:
            elapsed = agent.time - self.time

        if elapsed < 0.15:
            agent.controller.jump = True
        elif elapsed >= 0.15 and self.counter < 3:
            agent.controller.jump = False
            self.counter += 1
        elif elapsed < 0.3 or (not self.cancel and elapsed < 0.9):
            agent.controller.jump = True
            agent.controller.pitch = self.pitch
            agent.controller.yaw = self.yaw
        elif manual:
            return True
        else:
            agent.pop()
            agent.push(recovery())


class brake:
    @staticmethod
    def run(agent, manual=False):
        # current forward velocity
        speed = agent.me.local_velocity().x
        if abs(speed) > 10:
            # apply our throttle in the opposite direction
            # Once we're below a velocity of 50uu's, start to ease the throttle
            agent.controller.throttle = cap(speed / -50, -1, 1)
        elif not manual:
            agent.pop()


class goto:
    # Drives towards a designated (stationary) target
    # Optional vector controls where the car should be pointing upon reaching the target
    def __init__(self, target, vector=None, brake=False):
        self.target = target
        self.vector = vector
        self.brake = brake
        self.rule1_timer = -1

    def run(self, agent, manual=False):
        car_to_target = self.target - agent.me.location
        distance_remaining = car_to_target.flatten().magnitude()
        angle_to_target = abs(Vector(x=1).angle2D(agent.me.local(car_to_target)))
        direction = 1 if angle_to_target <= 2 or (agent.gravity.z > -450 and distance_remaining >= 1000) else -1

        agent.dbg_2d(f"Angle to target: {angle_to_target}")
        agent.dbg_2d(f"Distance to target: {distance_remaining}")
        agent.line(self.target - Vector(z=500), self.target + Vector(z=500), (255, 0, 255))

        if (not self.brake and distance_remaining < 350) or (self.brake and distance_remaining * 0.95 < (agent.me.local_velocity().x ** 2 * -1) / (2 * brake_accel.x)):
            if not manual:
                agent.pop()

            if self.brake:
                agent.push(brake())
            return

        final_target = self.target.copy()

        # Some adjustment to the final target to ensure it's inside the field and we don't try to drive through any goalposts to reach it
        if abs(agent.me.location.y) > 5120 - (agent.me.hitbox.length / 2):
            final_target.x = cap(final_target.x, -750, 750)

        if self.vector is not None:
            # See comments for adjustment in jump_shot for explanation
            side_of_vector = sign(self.vector.cross(Vector(z=1)).dot(car_to_target))
            car_to_target_perp = car_to_target.cross(Vector(z=side_of_vector)).normalize()
            adjustment = car_to_target.angle2D(self.vector) * distance_remaining / 3.14
            final_target += car_to_target_perp * adjustment

        local_target = agent.me.local_location(final_target)

        defaultPD(agent, local_target)
        target_speed = 2300 if distance_remaining > 640 else 1400
        defaultThrottle(agent, target_speed * direction)

        # this is to break rule 1's with TM8'S ONLY
        # 251 is the distance between center of the 2 longest cars in the game, with a bit extra
        if len(agent.friends) > 0 and agent.me.local_velocity().x < 50 and agent.controller.throttle == 1 and min(agent.me.location.flat_dist(car.location) for car in agent.friends) < 251:
            if self.rule1_timer == -1:
                self.rule1_timer = agent.time
            elif agent.time - self.rule1_timer > 1.5:
                agent.push(flip(Vector(y=250)))
                return
        elif self.rule1_timer != -1:
            self.rule1_timer = -1

        if distance_remaining < 320:
            agent.controller.boost = False
        agent.controller.handbrake = angle_to_target > 1.54 if direction == 1 else angle_to_target < 2.2

        velocity = agent.me.local_velocity().x
        if agent.boost_amount != 'unlimited' and angle_to_target < 0.03 and distance_remaining > 1920 and velocity > 600 and velocity < 2150 and distance_remaining / velocity > 2:
            if agent.gravity.z < -450:
                agent.push(wave_dash())
            else:
                agent.push(flip(local_target))
        elif direction == -1 and distance_remaining > 1000 and velocity < 200:
            agent.push(flip(local_target, True))
        elif agent.me.airborne:
            agent.push(recovery(self.target))


class shadow:
    def __init__(self):
        self.goto = goto(Vector(), brake=True)

    def run(self, agent):
        ball_slice = agent.ball_prediction_struct.slices[agent.future_ball_location_slice].physics.location
        ball_loc = Vector(ball_slice.x, ball_slice.y)
        agent.line(ball_loc, ball_loc + Vector(z=185), agent.renderer.white())
        ball_loc.y *= side(agent.team)

        if ball_loc.y < -2560 or (ball_loc.y < agent.ball.location.y * side(agent.team)):
            ball_loc = Vector(agent.ball.location.x, agent.ball.location.y * side(agent.team) + 640)

        distance = 1280

        target = Vector(y=(ball_loc.y + distance) * side(agent.team))
        agent.line(target, target + Vector(z=642), (255, 0, 255))

        target.x = (abs(ball_loc.x) + (250 if target.y < -1280 else -(1024 if abs(ball_loc.x) > 1024 else ball_loc.x))) * sign(ball_loc.x)

        self_to_target = agent.me.location.dist(target)

        if self_to_target < 250 and ball_loc.y < -640 and agent.me.velocity.magnitude() < 100 and abs(Vector(x=1).angle2D(agent.me.local_location(agent.ball.location))) > 0.1:
            agent.push(face_target(ball=True))
        else:
            self.goto.target = target
            self.goto.vector = agent.ball.location
            self.goto.run(agent, manual=True)

            if self_to_target < 500:
                agent.controller.boost = False


class retreat:
    def __init__(self):
        self.goto = goto(Vector())
        self.brake = brake()

    def run(self, agent):
        ball_slice = agent.ball_prediction_struct.slices[agent.future_ball_location_slice].physics.location
        ball = Vector(ball_slice.x, cap(ball_slice.y, -5100, 5100))
        agent.line(ball, ball + Vector(z=185), agent.renderer.white())
        ball.y *= side(agent.team)

        if ball.y < agent.ball.location.y * side(agent.team):
            ball = Vector(agent.ball.location.x, agent.ball.location.y * side(agent.team) + 640)

        target = self.get_target(agent)
        agent.line(target, target + Vector(z=642), (255, 0, 255))

        if target.flat_dist(agent.me.location) < 350:
            if agent.me.velocity.magnitude() > 100:
                self.brake.run(agent, manual=True)
            elif abs(Vector(x=1).angle2D(agent.me.local_location(ball))) > 0.5:
                agent.pop()
                agent.push(face_target(ball=True))
            else:
                agent.pop()
        else:
            self.goto.target = target
            self.goto.run(agent, manual=True)

    def get_target(self, agent):
        target = None
        ball_slice = agent.ball_prediction_struct.slices[agent.future_ball_location_slice].physics.location
        ball = Vector(ball_slice.x, ball_slice.y, ball_slice.z)
        ball_y = ball.y * side(agent.team)

        team_to_ball = [car.location.flat_dist(ball) for car in agent.friends if car.location.y * side(agent.team) >= ball_y - 50 and abs(car.location.x) < abs(ball.x)]
        self_to_ball = agent.me.location.flat_dist(ball)
        team_to_ball.append(self_to_ball)
        team_to_ball.sort()

        if agent.me.location.y * side(agent.team) >= ball_y - 50 and abs(agent.me.location.x) < abs(ball.x):
            if len(agent.friends) == 0 or abs(ball.x) < 900 or team_to_ball[-1] is self_to_ball:
                target = agent.friend_goal.location
            elif team_to_ball[0] is self_to_ball:
                target = agent.friend_goal.right_post if abs(ball.x) > 10 else agent.friend_goal.left_post

        if target is None:
            if len(agent.friends) <= 1:
                target = agent.friend_goal.location
            else:
                target = agent.friend_goal.left_post if abs(ball.x) > 10 else agent.friend_goal.right_post

        target = target.copy()
        target.y += 250 * side(agent.team) if len(agent.friends) == 0 or abs(ball.x) < 900 or team_to_ball[-1] is self_to_ball else -245 * side(agent.team)

        return target.flatten()


class face_target:
    def __init__(self, target=None, ball=False):
        self.target = target
        self.ball = ball
        self.start_loc = None
        self.counter = 0

    def run(self, agent):
        if self.ball:
            target = (agent.ball.location - agent.me.location).flatten()
        else:
            target = agent.me.velocity.flatten() if self.target is None else (self.target - agent.me.location).flatten()

        if agent.gravity.z < -550 and agent.gravity.z > -750:
            if self.counter == 0 and abs(Vector(x=1).angle(target)) <= 0.05:
                agent.pop()
                return

            if self.counter < 3:
                self.counter += 1

            target = agent.me.local(target)
            if self.counter < 3:
                agent.controller.jump = True
            elif agent.me.airborne and abs(Vector(x=1).angle(target)) > 0.05:
                defaultPD(agent, target)
            else:
                agent.pop()
        else:
            target = agent.me.local(target)
            angle_to_target = abs(Vector(x=1).angle2D(target))
            if angle_to_target > 0.1:
                if self.start_loc is None:
                    self.start_loc = agent.me.location

                direction = -1 if angle_to_target < 1.57 else 1

                agent.controller.steer = cap(target.y / 100, -1, 1) * direction
                agent.controller.throttle = 0.5 * direction
                agent.controller.handbrake = True
            else:
                agent.pop()
                if self.start_loc is not None:
                    agent.push(goto(self.start_loc, target, True))


class goto_boost:
    # very similar to goto() but designed for grabbing boost
    def __init__(self, boost):
        self.boost = boost
        self.goto = goto(self.boost.location)

    def run(self, agent):
        if not self.boost.large:
            self.goto.vector = agent.ball.location

        if not self.boost.active:
            agent.pop()
        else:
            self.goto.run(agent, manual=True)


class jump_shot:
    # Hits a target point at a target time towards a target direction
    # Target must be no higher than 300uu unless you're feeling lucky
    def __init__(self, intercept_time, shot_vector):
        self.intercept_time = intercept_time
        self.shot_vector = shot_vector
        self.ball_location = None
        # Flags for what part of the routine we are in
        self.jumping = False
        self.dodging = False
        self.counter = 0
        self.jump_time = -1

    def update(self, shot):
        self.intercept_time = shot.intercept_time
        self.shot_vector = shot.shot_vector

    def run(self, agent):
        if not agent.shooting:
            agent.shooting = True

        T = self.intercept_time - agent.time
        # Capping T above 0 to prevent division problems
        time_remaining = cap(T, 0.000001, 6)

        slice_n = round(T * 60)
        agent.dbg_2d(f"Shot slice #: {slice_n}")

        if T > 0.3 or self.ball_location is None:
            ball = agent.ball_prediction_struct.slices[slice_n].physics.location

            self.ball_location = Vector(ball.x, ball.y, ball.z)
            self.dodge_point = self.ball_location - (self.shot_vector * agent.best_shot_value)

            if self.dodge_point.z > 300:
                agent.pop()
                return

        car_to_ball = self.ball_location - agent.me.location
        # whether we should go forwards or backwards
        angle_to_target = abs(Vector(x=1).angle2D(agent.me.local(car_to_ball)))
        # whether we are to the left or right of the shot vector
        side_of_shot = sign(self.shot_vector.cross(Vector(z=1)).dot(car_to_ball))

        final_target = self.dodge_point.copy()

        # Some adjustment to the final target to ensure it's inside the field and we don't try to drive through any goalposts to reach it
        if abs(agent.me.location.y) > 5120 - (agent.me.hitbox.length / 2):
            final_target.x = cap(final_target.x, -750, 750)

        car_to_dodge_point = final_target - agent.me.location
        car_to_dodge_perp = car_to_dodge_point.cross(Vector(z=side_of_shot))  # perpendicular

        acceleration_required = backsolve(self.dodge_point, agent.me, time_remaining, Vector() if not self.jumping else agent.gravity)
        local_acceleration_required = agent.me.local(acceleration_required)
        distance_remaining = car_to_dodge_point.flatten().magnitude()
        # The adjustment causes the car to circle around the dodge point in an effort to line up with the shot vector
        # The adjustment slowly decreases to 0 as the bot nears the time to jump
        adjustment = car_to_dodge_point.angle2D(self.shot_vector) * distance_remaining / 2  # size of adjustment
        # controls how soon car will jump based on acceleration required
        # we set this based on the time remaining
        # bigger = later, which allows more time to align with shot vector
        # smaller = sooner
        jump_threshold = cap(T * 200, 250, 584)
        # factoring in how close to jump we are
        adjustment *= (cap(jump_threshold - (acceleration_required.z), 0, jump_threshold) / jump_threshold)
        # we don't adjust the final target if we are already jumping
        final_target += ((car_to_dodge_perp.normalize() * adjustment) if not self.jumping else 0) + Vector(z=50)

        distance_remaining = (final_target - agent.me.location).flatten().magnitude()
        direction = 1 if angle_to_target < 2.1 or (agent.gravity.z > -450 and distance_remaining >= 1000) else -1
        local_final_target = agent.me.local_location(final_target)

        # drawing debug lines to show the dodge point and final target (which differs due to the adjustment)
        agent.line(agent.me.location, self.dodge_point, agent.renderer.white())
        agent.line(self.dodge_point-Vector(z=100), self.dodge_point+Vector(z=100), agent.renderer.green())
        agent.line(final_target-Vector(z=100), final_target+Vector(z=100), agent.renderer.blue())

        vf = agent.me.velocity + agent.gravity * T
        xf = agent.me.location + agent.me.velocity * T + 0.5 * agent.gravity * T * T

        ball_l = agent.me.local_location(agent.ball.location)
        speed_required = 2300 if (self.ball_location.z < 92 + agent.me.hitbox.height / 2 and abs(ball_l.y) < 46 and ball_l.x > agent.me.hitbox.length) or distance_remaining > 2560 else distance_remaining * 0.975 / time_remaining
        agent.dbg_2d(f"Speed required: {speed_required}")

        if not self.jumping:
            defaultPD(agent, local_final_target)
            defaultThrottle(agent, speed_required * direction)

            agent.controller.handbrake = angle_to_target > 1.54 if direction == 1 else angle_to_target < 2.2

            local_dodge_point = agent.me.local_location(self.dodge_point.flatten())

            # use algebra to get the required jump time
            min_jump_time = self.dodge_point - agent.me.location
            min_jump_time -= agent.me.velocity * T
            min_jump_time -= 0.5 * agent.gravity * T * T
            min_jump_time -= agent.me.up * jump_speed * T
            min_jump_time /= jump_acc

            # This leaves us with xf = dT - 0.5d^2, so we use the quadratic formula to solve it
            min_jump_time = max(quadratic(-0.5, T, sum(min_jump_time) / 3))

            min_jump_time += 0.05

            f_vf = vf + (agent.me.up * (jump_speed + jump_acc * min_jump_time))
            local_vf = agent.me.local(f_vf.flatten())

            velocity = agent.me.local_velocity().x
            if T <= 0 or not virxrlcu.jump_shot_is_viable(T + 0.3, agent.boost_accel, agent.me.location.dist(self.ball_location), self.shot_vector.tuple(), agent.me.forward.tuple(), agent.me.boost if agent.boost_amount != 'unlimited' else 100000, velocity):
                # If we're out of time or not fast enough to be within 45 units of target at the intercept time, we pop
                agent.pop()
                agent.shooting = False
                agent.shot_weight = -1
                agent.shot_time = -1
                if agent.me.airborne:
                    agent.push(recovery())
            elif self.dodge_point.z > 92 + agent.me.hitbox.height / 2 and ((T <= min_jump_time and self.dodge_point.z >= 175) or (T <= 0.3 and self.dodge_point.z < 175)) and abs(local_dodge_point.y) < (92 if find_slope(self.shot_vector, car_to_ball) > 2 or self.dodge_point.z <= 92 + agent.me.hitbox.height / 2 else math.ceil(92 + agent.me.hitbox.width / 2)) and local_vf.x >= local_dodge_point.x:
                # Switch into the jump when the upward acceleration required reaches our threshold, and our lateral acceleration is negligible
                self.jumping = True
            elif agent.boost_amount != 'unlimited' and angle_to_target < 0.03 and velocity > 600 and velocity < speed_required - 150 and distance_remaining / velocity > 3:
                if agent.gravity.z < -450:
                    agent.push(wave_dash())
                else:
                    agent.push(flip(local_final_target))
            elif direction == -1 and velocity < 200 and distance_remaining / abs(velocity) > 2:
                agent.push(flip(local_final_target, True))
            elif agent.me.airborne:
                agent.push(recovery(local_final_target))
        else:
            if self.jump_time == -1:
                self.jump_time = agent.time

            jump_elapsed = agent.time - self.jump_time

            tau = jump_max_duration - jump_elapsed

            if jump_elapsed == 0:
                vf += agent.me.up * jump_speed
                xf += agent.me.up * jump_speed * T

            vf += agent.me.up * jump_acc * tau
            xf += agent.me.up * jump_acc * tau * (T - 0.5 * tau)

            delta_x = self.dodge_point - xf
            d_direction = delta_x.normalize()

            if abs(agent.me.forward.dot(d_direction)) > 0.5 and self.counter < 3:
                delta_v = delta_x.dot(agent.me.forward) / T
                if agent.me.boost > 0 and delta_v >= agent.boost_accel * min_boost_time:
                    agent.controller.boost = True
                else:
                    agent.controller.throttle = cap(delta_v / (throttle_accel * min_boost_time), -1, 1)

            if T <= -0.4 or (not agent.me.airborne and self.counter >= 3):
                agent.pop()
                agent.shooting = False
                agent.shot_weight = -1
                agent.shot_time = -1
                agent.push(recovery())
                return
            else:
                if self.counter == 3 and agent.me.location.dist(self.dodge_point) < (92.75 + agent.me.hitbox.length) * 1.05:
                    # Get the required pitch and yaw to flip correctly
                    vector = agent.me.local(self.shot_vector)
                    self.p = abs(vector.x) * -sign(vector.x)
                    self.y = abs(vector.y) * sign(vector.y) * direction

                    # simulating a deadzone so that the dodge is more natural
                    self.p = cap(self.p, -1, 1) if abs(self.p) > 0.1 else 0
                    self.y = cap(self.y, -1, 1) if abs(self.y) > 0.1 else 0

                    agent.controller.pitch = self.p
                    agent.controller.yaw = self.y
                    # Wait 1 more frame before dodging
                    self.counter += 1
                elif self.counter == 4:
                    # Dodge
                    agent.controller.jump = True
                    agent.controller.pitch = self.p
                    agent.controller.yaw = self.y
                else:
                    # Face the direction we're heading in, with the z of our target
                    target = agent.me.local(agent.me.velocity.flatten())
                    target.z = agent.me.local_location(self.dodge_point).z
                    defaultPD(agent, target)

                if jump_elapsed < jump_max_duration and vf.z < self.dodge_point.z:
                    # Initial jump to get airborne + we hold the jump button for extra power as required
                    agent.controller.jump = True
                elif self.counter < 3:
                    # Make sure we aren't jumping for at least 3 frames
                    self.counter += 1

        l_vf = vf + agent.me.location
        agent.line(l_vf-Vector(z=100), l_vf+Vector(z=100), agent.renderer.red())


class generic_kickoff:
    def __init__(self):
        self.start_time = -1
        self.flip = False

    def run(self, agent):
        if self.start_time == -1:
            self.start_time = agent.time

        if self.flip or agent.time - self.start_time > 3:
            agent.kickoff_done = True
            agent.pop()
            return

        target = agent.ball.location + Vector(y=(200 if agent.gravity.z < -600 and agent.gravity.z > -700 else 50)*side(agent.team))
        local_target = agent.me.local_location(target)

        defaultPD(agent, local_target)
        agent.controller.throttle = 1
        agent.controller.boost = True

        distance = local_target.magnitude()

        if distance < 550:
            self.flip = True
            agent.push(flip(agent.me.local_location(agent.foe_goal.location)))


class recovery:
    # Point towards our velocity vector and land upright, unless we aren't moving very fast
    # A vector can be provided to control where the car points when it lands
    def __init__(self, target=None):
        self.target = target

    def run(self, agent):
        local_target = agent.me.local(agent.me.velocity.flatten()) if self.target is None else agent.me.local((self.target - agent.me.location).flatten())

        agent.dbg_2d(f"Recovering towards {self.target}")
        recover_on_ceiling = agent.me.velocity.z > 0 and agent.me.location.z > 1500
        if recover_on_ceiling:
            agent.dbg_2d(f"Recovering on the ceiling")

        defaultPD(agent, local_target, upside_down=recover_on_ceiling)
        agent.controller.throttle = 1
        if not agent.me.airborne:
            agent.pop()


class ball_recovery:
    def __init__(self):
        self.recovery = recovery()

    def run(self, agent):
        self.recovery.target = agent.ball.location
        self.recovery.target.y = cap(self.recovery.target.y, -5100, 5100)
        self.recovery.run(agent)


class short_shot:
    # This routine drives towards the ball and attempts to hit it towards a given target
    # It does not require ball prediction and kinda guesses at where the ball will be on its own
    def __init__(self, target):
        self.target = target
        self.start_time = None

    def run(self, agent):
        if not agent.shooting:
            agent.shooting = True

        if self.start_time is None:
            self.start_time = agent.time

        car_to_ball, distance = (agent.ball.location - agent.me.location).normalize(True)
        ball_to_target = (self.target - agent.ball.location).normalize()
        angle_to_target = abs(Vector(x=1).angle2D(agent.me.local(car_to_ball)))

        relative_velocity = car_to_ball.dot(agent.me.velocity-agent.ball.velocity)
        eta = cap(distance / cap(relative_velocity, 400, 1400), 0, 1.5) if relative_velocity != 0 else 1.5

        # If we are approaching the ball from the wrong side the car will try to only hit the very edge of the ball
        left_vector = car_to_ball.cross(Vector(z=1))
        right_vector = car_to_ball.cross(Vector(z=-1))
        target_vector = -ball_to_target.clamp2D(left_vector, right_vector)
        final_target = agent.ball.location + (target_vector*(distance/2))

        # Some adjustment to the final target to ensure we don't try to drive through any goalposts to reach it
        if abs(agent.me.location.y) > 5130:
            final_target.x = cap(final_target.x, -750, 750)

        agent.line(final_target-Vector(z=100), final_target + Vector(z=100), (255, 255, 255))

        angles = defaultPD(agent, agent.me.local_location(final_target))
        defaultThrottle(agent, 1400)

        agent.controller.handbrake = angle_to_target > 1.54

        if abs(angles[1]) < 0.05 and (eta < 0.45 or distance < 150):
            agent.pop()
            agent.shooting = False
            agent.push(flip(agent.me.local(car_to_ball)))
        elif agent.time - self.start_time > 0.24:  # This will run for 3 ticks, then pop
            agent.pop()
            agent.shooting = False


class boost_down:
    def __init__(self):
        self.face = ball_recovery()

    def run(self, agent):
        if agent.me.boost == 0:
            agent.pop()
            agent.push(self.face)

        target = agent.me.local(agent.me.forward.flatten()*100 - Vector(z=100))
        defaultPD(agent, target)
        if not agent.me.airborne:
            agent.pop()
        elif abs(Vector(x=1).angle(target)) < 0.5:
            agent.controller.boost = True
