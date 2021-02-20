import numpy as np
from bpdb import set_trace

from pntddf.clock import Clock
from pntddf.dynamics import Dynamics
from pntddf.estimator import Estimator
from pntddf.radio import Radio
from pntddf.sensors import Sensors


class Agent:
    def __init__(self, env, name):
        self.env = env
        self.name = name

        self.index = env.AGENT_NAMES.index(name)

        self.config = self.env.agent_configs[name]

    def init(self):
        # Clock
        self.clock = Clock(self.env, self)

        # Radio
        self.radio = Radio(self.env, self)

        # Sensors
        self.sensors = Sensors(self.env, self)

        # Estimator
        self.estimator = Estimator(self.env, self)

    def __repr__(self):
        return str(self.name)
