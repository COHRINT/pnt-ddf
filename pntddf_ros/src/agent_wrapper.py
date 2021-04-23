#!/usr/bin/env python3

import numpy as np
import rospy
from pntddf.agent import Agent
from pntddf.env import setup_env


class Agent_Wrapper:
    def __init__(self):
        agent_label = rospy.get_namespace()  # e.g. '/agent_T/'
        self.agent_label = agent_label.replace("/", "")  # e.g. 'agent_T'
        self.agent_name = self.agent_label.split("_")[1]  # e.g. 'T'

        rospy.init_node(self.agent_label, log_level=rospy.INFO)

        config_file = "/opt/pnt_ddf_ws/config/sim.config"

        env = setup_env(config_file, ros=True)

        self.agent = env.agent_dict[self.agent_name]
        self.agent.init()

        rate = rospy.Rate(0.1)

        while not rospy.is_shutdown():
            rospy.loginfo("success {}".format(self.agent_name))

            rate.sleep()


if __name__ == "__main__":
    Agent_Wrapper()
