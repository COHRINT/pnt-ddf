import numpy as np
from bpdb import set_trace

from pntddf.asset_detections import Asset_Detections_Receiver
from pntddf.clock import Clock
from pntddf.dynamics import Dynamics
from pntddf.estimator import Estimator
from pntddf.gps import GPS
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

        # Estimator
        self.estimator = Estimator(self.env, self)

        # Asset Detections
        if self.env.ros:
            import rospy

            rospy.loginfo("ADR {}".format(self.name))
            self.asset_detections_receiver = Asset_Detections_Receiver(self.env, self)

        # Radio
        self.radio = Radio(self.env, self)

        # GPS
        if self.config.getboolean("gps") and not self.env.ros:
            self.gps = GPS(self.env, self)

    def __repr__(self):
        return str(self.name)
