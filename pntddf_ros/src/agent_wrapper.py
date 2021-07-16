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

        # register shutdown handle
        rospy.on_shutdown(self.shutdown)

        self.agent.init()

    def shutdown(self):
        try:
            df_state = self.agent.estimator.get_state_log_df()
            df_state.to_pickle("/opt/rosbags/df_state_{}.pkl".format(self.agent_name))
        except:
            pass

        df_meas = self.agent.estimator.get_residuals_log_df()
        df_meas.to_pickle("/opt/rosbags/df_meas_{}.pkl".format(self.agent_name))


if __name__ == "__main__":
    Agent_Wrapper()
