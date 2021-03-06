import numpy as np
from bpdb import set_trace

from pntddf.measurements import Asset_Detection


class Asset_Detections_Receiver:
    def __init__(self, env, agent):
        global rospy
        import rospy

        global DetectedAssetPosition
        global DetectedAssetPositionList
        from pntddf_ros.msg import (DetectedAssetPosition,
                                    DetectedAssetPositionList)

        self.env = env
        self.agent = agent

        rospy.Subscriber(
            "/detected_asset_test",
            DetectedAssetPositionList,
            self.detected_asset_callback,
        )

    def detected_asset_callback(self, msg):
        for detection in msg.assetPositions:
            try:
                agent_name = detection.id.split("_")[1]
            except:
                continue
            if agent_name in self.env.AGENT_NAMES:
                self.process_detection(agent_name, detection)

    def process_detection(self, agent_name, detection):
        t_receive = self.agent.clock.time()

        dim_names = ["x", "y", "z"]

        for d in range(self.env.n_dim):
            dim_name = dim_names[d]

            z = getattr(detection, dim_name)
            var = getattr(detection, "var_" + dim_name)

            agent = self.env.agent_dict[agent_name]

            measurement = Asset_Detection(self.env, z, var, d, agent, t_receive)

            self.agent.estimator.new_measurement(measurement)
