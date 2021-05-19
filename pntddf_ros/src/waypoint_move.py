#!/usr/bin/env python3

import numpy as np
import rospy
from pntddf.agent import Agent
from pntddf.env import setup_env


class WaypointMove:
    def __init__(self):
        agent_label = rospy.get_namespace()  # e.g. '/agent_T/'
        self.agent_label = agent_label.replace("/", "")  # e.g. 'agent_T'
        self.agent_name = self.agent_label.split("_")[1]  # e.g. 'T'

        rospy.init_node(self.agent_label+"_move", log_level=rospy.INFO)


        # config_file = "/opt/pnt_ddf_ws/config/sim2.config"
        config_file = rospy.get_param("~config_file")

        self.env = setup_env(config_file, ros=True)

        self.run()

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.env.dynamics.rover_dict[self.agent_name].move_to_waypoint()
            rate.sleep()



if __name__ == "__main__":
    WaypointMove()