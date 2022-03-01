import virx_erlu_rlib as rlru

from util.agent import BaseRoutine, Vector, VirxERLU


class ground_shot(BaseRoutine):
    def __init__(self, intercept_time: float, target: int):
        self.intercept_time = intercept_time
        self.target = target

    def run(self, agent: VirxERLU):
        pass

    def pop(self, agent: VirxERLU):
        rlru.remove_target(self.target)

