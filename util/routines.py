from util.utils import (Vector, backsolve, cap, defaultPD, defaultThrottle,
                        shot_valid, side, sign)

max_speed = 2300
throttle_accel = 100 * (2/3)
brake_accel = Vector(x=-3500)
boost_per_second = 33 + (1/3)
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

        # Some adjustment to the final target to ensure it's inside the field and we don't try to drive through any goalposts to reach it
        if abs(agent.me.location.y) > 5150:
            target.x = cap(target.x, -750, 750)

        local_target = agent.me.local(target - agent.me.location)

        angles = defaultPD(agent, local_target)
        defaultThrottle(agent, 1400)

        agent.controller.boost = False
        agent.controller.handbrake = True if abs(angles[1]) >= 2.3 or (agent.me.local_velocity().x >= 900 and abs(angles[1]) > 1.57) else agent.controller.handbrake

        velocity = 1+agent.me.velocity.magnitude()
        if distance_remaining < self.exit_distance:
            agent.pop()
            if self.exit_flip:
                agent.push(flip(local_target))
        elif abs(angles[1]) < 0.05 and velocity > 600 and velocity < 2150 and distance_remaining / velocity > 2:
            agent.push(flip(local_target))
        elif abs(angles[1]) > 2.8 and velocity < 200:
            agent.push(flip(local_target, True))
        elif agent.me.airborne:
            agent.push(recovery(target))


class wave_dash:
    def __init__(self):
        self.step = 0

    def run(self, agent):
        if self.step <= 5:
            self.step += 1
            agent.controller.pitch = 1
            agent.controller.yaw = agent.controller.role = 0

        if self.step <= 1:
            agent.controller.jump = True
        elif self.step <= 4:
            agent.controller.jump = False
        else:
            if (agent.me.location + (agent.me.velocity * 0.2)).z < 5:
                agent.controller.jump = True
                agent.controller.pitch = -1
                agent.controller.yaw = agent.controller.role = 0
                agent.pop()
            elif not agent.me.airborne:
                agent.pop()


class double_jump:
    def __init__(self, ball_location, intercept_time, shot_vector, best_shot_value):
        self.ball_location = ball_location
        self.intercept_time = intercept_time
        self.shot_vector = shot_vector
        self.dodge_point = self.ball_location - (self.shot_vector * best_shot_value)
        # Flags for what part of the routine we are in
        self.debug_start = True
        self.jumping = False
        self.dodged = False
        self.jump_time = -1
        self.counter = 0

    def run(self, agent):
        # This routine is the same as jump_shot, but it's designed to hit the ball above 250uus (and below 551uus or so) without any boost
        if not agent.shooting:
            agent.shooting = True

        if self.debug_start:
            agent.print(f"Hit ball via double jump {round(agent.me.location.dist(self.ball_location), 4)}uus away in {round(self.intercept_time - agent.time, 4)}s")
            self.debug_start = False

        raw_time_remaining = self.intercept_time - agent.time
        # Capping raw_time_remaining above 0 to prevent division problems
        time_remaining = cap(raw_time_remaining, 0.001, 10)
        car_to_ball = self.ball_location - agent.me.location
        # whether we should go forwards or backwards
        angle_to_target = abs(Vector(x=1).angle2D(agent.me.local_location(self.ball_location)))
        direction = 1 if angle_to_target < 2.2 else -1
        # whether we are to the left or right of the shot vector
        side_of_shot = sign(self.shot_vector.cross(Vector(z=1)).dot(car_to_ball))

        car_to_dodge_point = self.ball_location - agent.me.location
        car_to_dodge_perp = car_to_dodge_point.cross(Vector(z=side_of_shot))  # perpendicular
        distance_remaining = car_to_dodge_point.magnitude()

        speed_required = distance_remaining / time_remaining
        acceleration_required = backsolve(self.ball_location, agent.me, time_remaining - 0.5, Vector() if not self.jumping else agent.gravity)
        local_acceleration_required = agent.me.local(acceleration_required)

        # The adjustment causes the car to circle around the dodge point in an effort to line up with the shot vector
        # The adjustment slowly decreases to 0 as the bot nears the time to jump
        adjustment = car_to_dodge_point.angle2D(self.shot_vector) * distance_remaining / 2  # size of adjustment
        # controls how soon car will jump based on acceleration required
        # bigger == later and smaller == sooner
        # Max 584, min 1
        # Any thing below 250 might cause it to whiff
        # Anything above 400 might cause it to not hit it on target
        # 551 is the highest point that this double jump routine can hit the ball at
        jump_threshold = cap(551 - self.ball_location.z, 250, 400)
        # factoring in how close to jump we are
        adjustment *= (cap(jump_threshold - (acceleration_required.z), 0, jump_threshold) / jump_threshold)
        # we don't adjust the final target if we are already jumping
        final_target = self.ball_location + ((car_to_dodge_perp.normalize() * adjustment) if not self.jumping else 0) + Vector(z=50)
        # Ensuring our target isn't too close to the sides of the field, where our car would get messed up by the radius of the curves

        # Some adjustment to the final target to ensure it's inside the field and we don't try to drive through any goalposts to reach it
        if abs(agent.me.location.y) > 5120 - (agent.me.hitbox.width / 2):
            final_target.x = cap(final_target.x, -750, 750)

        local_final_target = agent.me.local_location(final_target)

        # drawing debug lines to show the dodge point and final target (which differs due to the adjustment)
        agent.polyline((agent.me.location, self.ball_location, self.dodge_point), agent.renderer.white())
        agent.line(self.dodge_point-Vector(z=100), self.dodge_point+Vector(z=100), agent.renderer.green())
        agent.line(final_target-Vector(z=100), final_target+Vector(z=100), agent.renderer.red())

        if not self.jumping:
            defaultPD(agent, local_final_target)
            defaultThrottle(agent, speed_required * direction)
            agent.controller.handbrake = (angle_to_target >= 2.3 or (agent.me.local_velocity().x >= 900 and angle_to_target > 1.57)) and direction == 1

            velocity = 1+agent.me.velocity.magnitude()
            if raw_time_remaining <= 0 or (speed_required - 2300) * time_remaining > 45 or not shot_valid(agent, self):
                # If we're out of time or not fast enough to be within 45 units of target at the intercept time, we pop
                agent.pop()
                agent.shooting = False
                agent.shot_weight = -1
                agent.shot_time = -1
                if agent.me.airborne:
                    agent.push(recovery())
            elif local_acceleration_required.z > jump_threshold and local_acceleration_required.z > local_acceleration_required.flatten().magnitude() and angle_to_target < 0.1:
                # Switch into the jump when the upward acceleration required reaches our threshold, and our lateral acceleration is negligible
                self.jumping = True
            elif angle_to_target < 0.05 and distance_remaining > (2560 if velocity < 1400 else (3840 if velocity < 2100 else 5120)) and velocity > 600 and velocity < speed_required - 150 and distance_remaining / velocity > 2:
                agent.push(flip(local_final_target))
            elif angle_to_target >= 2 and distance_remaining > 1000 and velocity < 200 and distance_remaining / velocity > 2:
                agent.push(flip(local_final_target, True))
            elif agent.me.airborne:
                agent.push(recovery(local_final_target))
        else:
            jump_elapsed = agent.time - self.jump_time

            if (raw_time_remaining > 0.2 and not shot_valid(agent, self, 150)) or raw_time_remaining <= -0.4 or (not agent.me.airborne and self.counter > 0):
                agent.pop()
                agent.shooting = False
                agent.shot_weight = -1
                agent.shot_time = -1
                agent.push(recovery())
            elif self.counter == 0 and (self.jump_time == -1 or jump_elapsed < jump_max_duration):
                if local_acceleration_required.z > 0 and raw_time_remaining > 0.083:
                    if self.jump_time == -1:
                        self.jump_time = agent.time
                    # Initial jump to get airborne + we hold the jump button for extra power as required
                    agent.controller.jump = True
            elif self.counter < 3:
                # make sure we aren't jumping for at least 3 frames - in this time, we can start to adjust ourself, if it's needed
                defaultPD(agent, agent.me.local_location(self.dodge_point))
                self.counter += 1
            elif jump_elapsed < 0.3 and not self.dodged and local_acceleration_required.z > 0:
                agent.controller.jump = True
                self.dodged = True
            elif jump_elapsed >= 0.3 or self.dodged or local_acceleration_required.z < 0:
                defaultPD(agent, agent.me.local_location(self.dodge_point), upside_down=True)
                agent.controller.boost = abs(Vector(x=1).angle(agent.me.local(car_to_ball))) < 0.1 and agent.me.local_velocity().x < 2300 - (agent.boost_accel / 120)


class Aerial:
    def __init__(self, ball_intercept, intercept_time):
        self.target = ball_intercept
        self.intercept_time = intercept_time
        self.jumping = True
        self.ceiling = False
        self.time = -1
        self.jump_time = -1
        self.counter = 0

    def run(self, agent):
        if not agent.shooting:
            agent.shooting = True

        if self.time == -1:
            elapsed = 0
            self.time = agent.time
            agent.print(f"Hit ball via aerial {round(agent.me.location.dist(self.target), 4)}uus away in {round(self.intercept_time - self.time, 4)}s")
        else:
            elapsed = agent.time - self.time

        T = self.intercept_time - agent.time
        xf = agent.me.location + agent.me.velocity * T + 0.5 * agent.gravity * T ** 2

        if self.jumping and agent.me.location.z < 2044 - agent.me.hitbox.height * 1.1:
            agent.dbg_2d("Jumping")
            if self.jump_time == -1:
                self.jump_time = agent.time

            jump_elapsed = agent.time - self.jump_time

            tau = jump_max_duration - jump_elapsed

            if jump_elapsed == 0:
                xf += agent.me.up * jump_speed * T

            xf += agent.me.up * jump_acc * tau * (T - 0.5 * tau)
            xf += agent.me.up * jump_speed * (T - tau)

            if jump_elapsed < jump_max_duration:
                agent.controller.jump = True
            elif jump_elapsed >= jump_max_duration and self.counter < 3:
                self.counter += 1
            elif jump_elapsed < 0.3:
                agent.controller.jump = True
            else:
                self.jumping = jump_elapsed <= 0.3
        elif self.jumping:
            self.jumping = False
            self.ceiling = True
            self.target -= Vector(z=92)

        if self.ceiling:
            agent.dbg_2d(f"Ceiling shot")

        delta_x = self.target - xf
        direction = delta_x.normalize()

        agent.line(agent.me.location, self.target, agent.renderer.white())
        agent.line(self.target - Vector(z=100), self.target + Vector(z=100), agent.renderer.green())
        agent.line(agent.me.location, direction, agent.renderer.red())

        if not (jump_max_duration <= elapsed and elapsed < 0.3 and self.counter == 3):
            defaultPD(agent, agent.me.local(delta_x if delta_x.magnitude() > 50 else (self.target - agent.me.location)), upside_down=self.ceiling)

        if agent.me.forward.angle(direction) < 0.3:
            if delta_x.magnitude() > 50:
                agent.controller.boost = 1
            else:
                agent.controller.throttle = cap(0.5 * throttle_accel * T * T, 0, 1)

        still_valid = shot_valid(agent, self, threshold=250, target=self.target)

        if T <= 0 or not still_valid:
            if not still_valid:
                agent.print("Aerial is no longer valid")

            agent.pop()
            agent.shooting = False
            agent.shot_weight = -1
            agent.shot_time = -1
            agent.push(ball_recovery())
        elif self.ceiling and self.target.dist(agent.me.location) < 92 + agent.me.hitbox.length and not agent.me.doublejumped and agent.me.location.z < agent.ball.location.z + 92 and self.target.y * side(agent.team) > -4240:
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
        speed = agent.me.local_velocity().x
        if speed > 0:
            agent.controller.throttle = -1
            if speed < 25 and not manual:
                agent.pop()
        elif speed < 0:
            agent.controller.throttle = 1
            if speed > -25 and not manual:
                agent.pop()
        elif not manual:
            agent.pop()


class goto:
    # Drives towards a designated (stationary) target
    # Optional vector controls where the car should be pointing upon reaching the target
    def __init__(self, target, vector=None, brake=False):
        self.target = target
        self.vector = vector
        self.brake = brake

    def run(self, agent, manual=False):
        car_to_target = self.target - agent.me.location
        distance_remaining = car_to_target.flatten().magnitude()
        angle_to_target = abs(Vector(x=1).angle2D(agent.me.local(car_to_target)))
        direction = 1 if angle_to_target < 2.2 else -1

        agent.dbg_2d(f"Angle to target: {angle_to_target}")
        agent.dbg_2d(f"Distance to target: {distance_remaining}")
        agent.line(self.target - Vector(z=500), self.target + Vector(z=500), [255, 0, 255])

        if (not self.brake and distance_remaining < 350) or (self.brake and distance_remaining < (agent.me.local_velocity().x * 2 * -1) / (2 * brake_accel.x)):
            if not manual:
                agent.pop()

            if self.brake:
                agent.push(brake())
            return

        if self.vector is not None:
            # See comments for adjustment in jump_shot for explanation
            side_of_vector = sign(self.vector.cross(Vector(z=1)).dot(car_to_target))
            car_to_target_perp = car_to_target.cross(Vector(z=side_of_vector)).normalize()
            adjustment = car_to_target.angle2D(self.vector) * distance_remaining / 3.14
            final_target = self.target + (car_to_target_perp * adjustment)
        else:
            final_target = self.target

        # Some adjustment to the final target to ensure it's inside the field and we don't try to drive through any goalposts to reach it
        if abs(agent.me.location.y) > 5120 + (agent.me.hitbox.length / 2):
            final_target.x = cap(final_target.x, -750, 750)

        local_target = agent.me.local_location(final_target)

        defaultPD(agent, local_target)
        target_speed = 2300 if distance_remaining > 1280 else 1400
        defaultThrottle(agent, target_speed * direction)

        if len(agent.friends) > 0 and agent.me.local_velocity().x < 10 and agent.controller.throttle == 1 and min(agent.me.location.flat_dist(car.location) for car in agent.friends) < 251:
            agent.push(flip(Vector(y=250)))
            return

        if agent.me.boost < 30:
            agent.controller.boost = False
        agent.controller.handbrake = (angle_to_target >= 2.3 or (agent.me.local_velocity().x >= 900 and angle_to_target > 1.57)) and direction == 1

        velocity = 1+agent.me.velocity.magnitude()
        if angle_to_target < 0.05 and distance_remaining > 1920 and velocity > 600 and velocity < 2150 and distance_remaining / velocity > 2:
            agent.push(flip(local_target))
        elif direction == -1 and distance_remaining > 1000 and velocity < 200 and distance_remaining / velocity > 2:
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

        distance = 1280 if len(agent.friends) >= 2 else 2560

        target = Vector(y=(ball_loc.y + distance) * side(agent.team))
        agent.line(target, target + Vector(z=642), (255, 0, 255))

        target.x = (abs(ball_loc.x) + (250 if ball_loc.y < -640 else -750)) * sign(ball_loc.x)

        self_to_target = agent.me.location.dist(target)

        if self_to_target > 350:
            self.goto.target = target
            ball_loc.y *= side(agent.team)
            self.goto.vector = ball_loc
            self.goto.run(agent, manual=True)

            if self_to_target < 500:
                agent.controller.boost = False
                agent.controller.throttle = cap(agent.controller.throttle, -0.75, 0.75)


class retreat:
    def __init__(self):
        self.goto = goto(Vector())
        self.brake = brake()

    def run(self, agent):
        ball_slice = agent.ball_prediction_struct.slices[agent.future_ball_location_slice].physics.location
        ball = Vector(ball_slice.x, ball_slice.y)
        agent.line(ball, ball + Vector(z=185), agent.renderer.white())

        target = self.get_target(agent)
        agent.line(target, target + Vector(z=642), (255, 0, 255))

        if target.flat_dist(agent.me.location) < 350:
            if agent.me.velocity.magnitude() > 100:
                self.brake.run(agent)
            elif abs(Vector(x=1).angle2D(agent.me.local_location(ball))) > 0.25:
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
        self.counter = 0

    def run(self, agent):
        if self.counter == 0:
            agent.controller.jump = True
            self.counter += 1
        elif self.counter == 1:
            agent.pop()
            if self.ball:
                agent.push(ball_recovery())
            else:
                agent.push(recovery(self.target))


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
            self.goto.run(agent)


class jump_shot:
    # Hits a target point at a target time towards a target direction
    # Target must be no higher than 300uu unless you're feeling lucky
    def __init__(self, ball_location, intercept_time, shot_vector, best_shot_value):
        self.ball_location = ball_location
        self.intercept_time = intercept_time
        self.shot_vector = shot_vector
        self.dodge_point = self.ball_location - (self.shot_vector * best_shot_value)
        # Flags for what part of the routine we are in
        self.jumping = False
        self.dodging = False
        self.counter = 0

    def run(self, agent):
        if not agent.shooting:
            agent.shooting = True

        raw_time_remaining = self.intercept_time - agent.time
        # Capping raw_time_remaining above 0 to prevent division problems
        time_remaining = cap(raw_time_remaining, 0.001, 10)
        car_to_ball = self.ball_location - agent.me.location
        # whether we should go forwards or backwards
        angle_to_target = abs(Vector(x=1).angle2D(agent.me.local(car_to_ball)))
        direction = 1 if angle_to_target < 2.2 else -1
        # whether we are to the left or right of the shot vector
        side_of_shot = sign(self.shot_vector.cross(Vector(z=1)).dot(car_to_ball))

        car_to_dodge_point = self.dodge_point - agent.me.location
        car_to_dodge_perp = car_to_dodge_point.cross(Vector(z=side_of_shot))  # perpendicular
        distance_remaining = car_to_dodge_point.magnitude()

        speed_required = distance_remaining / time_remaining
        acceleration_required = backsolve(self.dodge_point, agent.me, time_remaining, Vector() if not self.jumping else agent.gravity)
        local_acceleration_required = agent.me.local(acceleration_required)

        # The adjustment causes the car to circle around the dodge point in an effort to line up with the shot vector
        # The adjustment slowly decreases to 0 as the bot nears the time to jump
        adjustment = car_to_dodge_point.angle2D(self.shot_vector) * distance_remaining / 2  # size of adjustment
        # controls how soon car will jump based on acceleration required
        # If we're angled off really far from the dodge point, then we'll delay the jump in an effort to get a more accurate shot
        # If we're dead on with the shot (ex 0.25 radians off) then we'll jump a lot sooner
        # any number larger than 0 works for the minimum
        # 584 is the highest you can go, for the maximum
        jump_threshold = cap(abs(Vector(x=1).angle2D(agent.me.local_location(self.dodge_point))) * 400, 100 if self.dodge_point.z > 150 else 200, 500)
        # factoring in how close to jump we are
        adjustment *= (cap(jump_threshold - (acceleration_required.z), 0, jump_threshold) / jump_threshold)
        # we don't adjust the final target if we are already jumping
        final_target = self.dodge_point + ((car_to_dodge_perp.normalize() * adjustment) if not self.jumping else 0) + Vector(z=50)
        # Ensuring our target isn't too close to the sides of the field, where our car would get messed up by the radius of the curves

        # Some adjustment to the final target to ensure it's inside the field and we don't try to drive through any goalposts to reach it
        if abs(agent.me.location.y) > 5120 - (agent.me.hitbox.width / 2):
            final_target.x = cap(final_target.x, -750, 750)

        local_final_target = agent.me.local_location(final_target)

        # drawing debug lines to show the dodge point and final target (which differs due to the adjustment)
        agent.line(agent.me.location, self.dodge_point, agent.renderer.white())
        agent.line(self.dodge_point-Vector(z=100), self.dodge_point+Vector(z=100), agent.renderer.red())
        agent.line(final_target-Vector(z=100), final_target+Vector(z=100), agent.renderer.green())

        if not self.jumping:
            defaultPD(agent, local_final_target)
            defaultThrottle(agent, speed_required * direction)

            agent.controller.handbrake = (angle_to_target >= 2.3 or (agent.me.local_velocity().x >= 900 and angle_to_target > 1.57)) and direction == 1

            velocity = 1+agent.me.velocity.magnitude()
            if raw_time_remaining <= 0 or (speed_required - 2300) * time_remaining > 45 or not shot_valid(agent, self):
                # If we're out of time or not fast enough to be within 45 units of target at the intercept time, we pop
                agent.pop()
                agent.shooting = False
                agent.shot_weight = -1
                agent.shot_time = -1
                if agent.me.airborne:
                    agent.push(recovery())
            elif local_acceleration_required.z > jump_threshold and local_acceleration_required.z > local_acceleration_required.flatten().magnitude():
                # Switch into the jump when the upward acceleration required reaches our threshold, and our lateral acceleration is negligible
                self.jumping = True
            elif angle_to_target < 0.05 and distance_remaining > (2560 if velocity < 1400 else (3840 if velocity < 2100 else 5120)) and velocity > 600 and velocity < speed_required - 150 and distance_remaining / velocity > 2:
                agent.push(flip(local_final_target))
            elif angle_to_target >= 2 and distance_remaining > 1000 and velocity < 200 and distance_remaining / velocity > 2:
                agent.push(flip(local_final_target, True))
            elif agent.me.airborne:
                agent.push(recovery(local_final_target))
        else:
            if (raw_time_remaining > 0.2 and not shot_valid(agent, self, 150)) or raw_time_remaining <= -0.4 or (not agent.me.airborne and self.counter > 0):
                agent.pop()
                agent.shooting = False
                agent.shot_weight = -1
                agent.shot_time = -1
                agent.push(recovery())
            elif self.counter == 0 and local_acceleration_required.z > 0 and raw_time_remaining > 0.083:
                # Initial jump to get airborne + we hold the jump button for extra power as required
                agent.controller.jump = True
            elif self.counter < 3:
                # make sure we aren't jumping for at least 3 frames
                agent.controller.jump = False
                self.counter += 1
            elif raw_time_remaining <= 0.1 and raw_time_remaining > -0.4:
                # dodge in the direction of the shot_vector
                agent.controller.jump = True
                if not self.dodging:
                    vector = agent.me.local(self.shot_vector)
                    self.p = abs(vector.x) * -sign(vector.x)
                    self.y = abs(vector.y) * sign(vector.y) * direction
                    self.dodging = True
                # simulating a deadzone so that the dodge is more natural
                agent.controller.pitch = self.p if abs(self.p) > 0.2 else 0
                agent.controller.yaw = self.y if abs(self.y) > 0.3 else 0


class generic_kickoff:
    def __init__(self):
        self.flip = False

    def run(self, agent):
        if self.flip:
            agent.kickoff_done = True
            agent.pop()
            return

        target = agent.ball.location + Vector(y=200*side(agent.team))
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
        self.recovery.run(agent)


class short_shot:
    # This routine drives towards the ball and attempts to hit it towards a given target
    # It does not require ball prediction and kinda guesses at where the ball will be on its own
    def __init__(self, target):
        self.target = target

    def run(self, agent):
        if not agent.shooting:
            agent.shooting = True

        car_to_ball, distance = (agent.ball.location - agent.me.location).normalize(True)
        ball_to_target = (self.target - agent.ball.location).normalize()

        relative_velocity = car_to_ball.dot(agent.me.velocity-agent.ball.velocity)
        if relative_velocity != 0:
            eta = cap(distance / cap(relative_velocity, 400, 2300), 0, 1.5)
        else:
            eta = 1.5

        # If we are approaching the ball from the wrong side the car will try to only hit the very edge of the ball
        left_vector = car_to_ball.cross(Vector(z=1))
        right_vector = car_to_ball.cross(Vector(z=-1))
        target_vector = -ball_to_target.clamp(left_vector, right_vector)
        final_target = agent.ball.location + (target_vector*(distance/2))

        # Some adjustment to the final target to ensure we don't try to drive through any goalposts to reach it
        if abs(agent.me.location.y) > 5130:
            final_target.x = cap(final_target.x, -750, 750)

        agent.line(final_target-Vector(z=100), final_target + Vector(z=100), [255, 255, 255])

        angles = defaultPD(agent, agent.me.local_location(final_target))
        defaultThrottle(agent, 2300 if distance > 1600 else 2300-cap(1600*abs(angles[1]), 0, 2050))
        agent.controller.boost = False
        agent.controller.handbrake = True if abs(angles[1]) >= 2.3 or (agent.me.local_velocity().x >= 900 and abs(angles[1]) > 1.57) else agent.controller.handbrake

        if abs(angles[1]) < 0.05 and (eta < 0.45 or distance < 150):
            agent.pop()
            agent.shooting = False
            agent.shot_weight = -1
            agent.shot_time = -1
            agent.push(flip(agent.me.local(car_to_ball)))
