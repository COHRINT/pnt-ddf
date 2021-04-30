#!/usr/bin/env python3

import numpy as np
import rospy
from pntddf_ros.msg import DetectedAssetPosition, DetectedAssetPositionList


class Asset_Pos_Publisher:
    def __init__(self):
        self.pub = rospy.Publisher(
            "detected_asset_test", DetectedAssetPositionList, queue_size=2
        )

        self.run()

    def generateRandomData(self):
        data = []
        for agent_name in ["T", "U", "V"]:
            asset = DetectedAssetPosition()

            asset.header.stamp = rospy.get_rostime()
            asset.header.frame_id = "asset_" + agent_name + "/base_link"

            asset.id = "jackal_" + agent_name
            asset.x = np.random.random() * 20 - 10
            asset.y = np.random.random() * 20 - 10
            asset.z = np.random.random() * 20 - 10

            asset.var_x = np.random.random()
            asset.var_y = np.random.random()
            asset.var_z = np.random.random()

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
