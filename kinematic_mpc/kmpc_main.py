# MIT License

# Copyright (c) Hongrui Zheng, Johannes Betz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Single Track Kinematic MPC waypoint tracker example

Author: Hongrui Zheng, Derek Zhou
"""

import numpy as np
import gym
import math

from kinematic_mpc import KMPCPlanner


def main():
    """
    STMPC example. This example uses fixed waypoints throughout the 2 laps.
    For an example using dynamic waypoints, see the lane switcher example.
    """

    # loading waypoints
    # waypoints = np.loadtxt('./levine_centerline.csv', delimiter=';', skiprows=3)
    waypoints = np.loadtxt('./levine_raceline.csv', delimiter=';', skiprows=3)  # levine raceline
    # waypoints = np.loadtxt('./Spielberg_raceline.csv', delimiter=';', skiprows=3)  # Spielberg

    # [x, y, yaw, v]
    # mpc_line = [waypoints[:, 1], waypoints[:, 2], waypoints[:, 3], waypoints[:, 5]]
    mpc_line = [waypoints[:, 1], waypoints[:, 2], waypoints[:, 3] + math.pi / 2, waypoints[:, 5]]  # levine raceline
    # mpc_line = [waypoints[:, 1], waypoints[:, 2], waypoints[:, 3], waypoints[:, 5]]  # Spielberg
    planner = KMPCPlanner(waypoints=mpc_line)

    # create environment
    env = gym.make('f110_gym:f110-v0', map='./levine_slam', map_ext='.pgm', num_agents=1)
    # obs, _, done, _ = env.reset(np.array([[2.51, 3.29, 1.58]]))
    obs, _, done, _ = env.reset(np.array([[2.28, 0.30, -0.67 + math.pi / 2]]))  # levine raceline
    # env = gym.make('f110_gym:f110-v0', map='./Spielberg_map', map_ext='.png', num_agents=1)  # Spielberg
    # obs, _, done, _ = env.reset(np.array([[0.0, -0.84, 3.40]]))

    laptime = 0.0
    up_to_speed = False
    while not done:
        if up_to_speed:
            steer, speed = planner.plan(env.sim.agents[0].state)
            print("steer = {}, speed = {}".format(round(steer, 5), speed))
            obs, timestep, done, _ = env.step(np.array([[steer, speed]]))
            laptime += timestep
            env.render(mode='human')
        else:
            steer = 0.0
            speed = 10.0
            print("0 & 10")
            # print("steer = {}, speed = {}".format(round(steer, 5), speed))
            obs, timestep, done, _ = env.step(np.array([[steer, speed]]))
            laptime += timestep
            env.render(mode='human')
            if obs['linear_vels_x'][0] > 0.1:
                up_to_speed = True
    print('Sim elapsed time:', laptime)


if __name__ == '__main__':
    main()
