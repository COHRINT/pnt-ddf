from bpdb import set_trace

from measurements import GPS_Measurement


class GPS:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.gps_rate = self.agent.config.getfloat("gps_rate")

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
        for d in range(self.env.n_dim):
            gps_measurement = GPS_Measurement(self.env, d, self.agent)
            gps_measurement.local = True

            self.agent.estimator.new_measurement(gps_measurement)
