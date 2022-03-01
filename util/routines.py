from util.agent import Vector, VirxERLU


class BaseRoutine:
    def run(self, agent: VirxERLU):
        raise NotImplementedError


class ground_shot(BaseRoutine):
    def __init__(self, target: Vector):
        self.target = target

    def run(self, agent: VirxERLU):
        pass

