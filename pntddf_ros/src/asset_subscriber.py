#!/usr/bin/env python3
import sys

from rospy import rostime
sys.path.append('../..')


import rospy
from pntddf.measurements import Measurement
from pntddf_ros.msg import DetectedAssetPosition, DetectedAssetPositionList
import numpy as np
from copy import copy



class DAP_Measurement(Measurement):
    def __init__(self):
        super().__init__()

        rospy.Subscriber("detected_asset_test",DetectedAssetPositionList,self.detected_asset_callback)



    def detected_asset_callback(self,msg):
        
        for i in range(len(msg.assetPositions)):
            print(msg.assetPositions[i])


if __name__ == "__main__":
    rospy.init_node("DAP_subscriber")
    dap = DAP_Measurement()
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        rate.sleep()

