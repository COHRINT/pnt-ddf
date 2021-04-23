import numpy as np
from bpdb import set_trace

from pntddf.measurements import GPS_Measurement


class GPS:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.gps_rate = self.agent.config.getfloat("gps_rate")

        self.sigma_gps = self.agent.config.getfloat("sigma_gps")

        if not self.env.ros:
            self.gps_process = self.env.process(self.gps())
        else:
            self.gps_ros()

    def gps(self):
        while True:
            self.report_gps_measurement()

            yield self.env.timeout(self.gps_rate)

    def gps_ros(self):
        pass

    def report_gps_measurement(self):
        t_receive = self.agent.clock.time()

        position = self.env.dynamics.get_true_position(self.agent.name)

        for d in range(self.env.n_dim):
            measurement = position[d]
            noise = np.random.normal(0, self.sigma_gps)

            z = measurement + noise
            gps_measurement = GPS_Measurement(self.env, z, d, self.agent, t_receive)

            self.agent.estimator.new_measurement(gps_measurement)
