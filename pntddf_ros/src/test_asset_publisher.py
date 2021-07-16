#!/usr/bin/env python3

import numpy as np
import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from pntddf_ros.msg import DetectedAssetPosition, DetectedAssetPositionList


class Asset_Pos_Publisher:
    def __init__(self):
        self.pub = rospy.Publisher(
            "detected_asset_test", DetectedAssetPositionList, queue_size=2
        )

        rospy.wait_for_service("/gazebo/get_model_state")
        self.get_model_srv = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )

        self.run()

    def generateRandomData(self):
        data = []
        for agent_name in ["T"]:
            asset = DetectedAssetPosition()

            asset.header.stamp = rospy.get_rostime()
            asset.header.frame_id = "asset_" + agent_name + "/base_link"

            asset.id = "jackal_" + agent_name

            model = GetModelStateRequest()
            model.model_name = "jackal_{}".format(agent_name)
            result = self.get_model_srv(model)

            asset.x = result.pose.position.x + np.random.normal(0, 1)
            asset.y = result.pose.position.y + np.random.normal(0, 1)
            asset.z = result.pose.position.z + np.random.normal(0, 1)

            asset.var_x = 1
            asset.var_y = 1
            asset.var_z = 1

            data.append(asset)
        return data

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.pub.publish(self.generateRandomData())
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("asset_pub_test")
    Asset_Pos_Publisher()
