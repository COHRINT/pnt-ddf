from bpdb import set_trace

from pntddf.clock import Clock
from pntddf.estimator import Estimator


class Agent_Centralized:
    def __init__(self, env, name):
        self.env = env
        self.name = name

        self.config = self.env.config_centralized

        self.agent_reference = "A"

    def init(self):
        self.clock = self.env.agent_dict[self.agent_reference].clock

        # Estimator
        self.estimator = Estimator(self.env, self)

    def __repr__(self):
        return str(self.name)
