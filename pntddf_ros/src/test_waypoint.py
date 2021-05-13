#!/usr/bin/env python3

import actionlib
import numpy as np
import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class Waypoint:
    def __init__(self):
        self.robot_name = "jackal_T"

        self.waypoint = np.array([10,10])

        self.client = actionlib.SimpleActionClient("/{}/move_base".format(self.robot_name),MoveBaseAction)
        self.client.wait_for_server()


        self.run()

    def run(self):
        while not rospy.is_shutdown():
            goal = MoveBaseGoal()

            goal.target_pose.header.frame_id = "{}/odom".format(self.robot_name)
            goal.target_pose.header.stamp = rospy.Time.now()

            goal.target_pose.pose.position.x = self.waypoint[0]
            goal.target_pose.pose.position.y = self.waypoint[1]

            goal.target_pose.pose.orientation.w = 1.0

            self.client.send_goal(goal)

            sleep_duration = 1.0

            rospy.sleep(sleep_duration)

            self.client.cancel_goal()

if __name__ == "__main__":
    rospy.init_node("waypoint_test")
    Waypoint()


