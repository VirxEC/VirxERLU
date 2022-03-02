from traceback import print_exc

import virx_erlu_rlib as rlru

from util.agent import BaseRoutine, Vector, VirxERLU
from util import utils


class ground_shot(BaseRoutine):
    def __init__(self, intercept_time: float, target: int):
        self.intercept_time = intercept_time
        self.target = target

    def update(self, intercept_time: float, target: int):
        rlru.remove_target(self.target)
        rlru.confirm_target(target)
        self.intercept_time = intercept_time
        self.target = target

    def run(self, agent: VirxERLU):
        future_ball_location = Vector(*rlru.get_slice(self.shot_time).location)

        agent.point(future_ball_location, self.renderer.purple())

        T = self.intercept_time - agent.time

        try:
            shot_info = rlru.get_data_for_shot_with_target(self.shot)
        except AssertionError:
            agent.pop()
            print_exc()
            return
        except ValueError:
            # We ran out of time
            agent.pop()
            return

        if len(shot_info.path_samples) > 2:
            agent.polyline(tuple((sample[0], sample[1], 30) for sample in shot_info.path_samples), agent.renderer.lime())
        else:
            agent.line(agent.me.location, shot_info.final_target, agent.renderer.lime())

        final_target = Vector(*shot_info.final_target)
        agent.point(final_target, self.renderer.red())
        distance_remaining = shot_info.distance_remaining

        speed_required = min(distance_remaining / T, 2300)
        local_final_target = agent.me.local_location(final_target.flatten())

        utils.defaultDrive(agent, speed_required, local_final_target)

    def pre_pop(self, agent: VirxERLU):
        rlru.remove_target(self.target)

