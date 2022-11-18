"""
    MEAM 517 Final Project - LQR Steering Speed Control - main application
    Author: Derek Zhou & Tancy Zhao
    References: https://f1tenth-gym.readthedocs.io/en/latest/index.html
                https://github.com/f1tenth/f1tenth_gym/tree/main/examples
                https://github.com/f1tenth/f1tenth_planning/tree/main/f1tenth_planning/control/lqr
                https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathTracking/lqr_steer_control
"""

import gym
import numpy as np
import yaml
import os

from mpc_steering_speed import Waypoint, MPCSteeringSpeedController
from log import xlsx_log_action, xlsx_log_observation
from render import Renderer


def main():
    # load map & yaml
    map_name = 'MoscowRaceway'  # Spielberg, example, MoscowRaceway, Catalunya
    map_path = os.path.abspath(os.path.join('..', 'map', map_name))
    yaml_config = yaml.load(open(map_path + '/' + map_name + '_map.yaml'), Loader=yaml.FullLoader)

    # load waypoints
    csv_data = np.loadtxt(map_path + '/' + map_name + '_raceline.csv', delimiter=';', skiprows=0)
    waypoints = Waypoint(map_name, csv_data)

    # load controller
    controller = MPCSteeringSpeedController(waypoints)

    # create env & init
    env = gym.make('f110_gym:f110-v0', map=map_path + '/' + map_name + '_map', map_ext='.png', num_agents=1)
    renderer = Renderer(waypoints)
    env.add_render_callback(renderer.render_waypoints)

    # init
    init_pos = np.array([yaml_config['init_pos']])
    obs, _, done, _ = env.reset(init_pos)
    lap_time = 0.0

    # placeholder for logging, plotting, and debugging
    log_action = []
    log_obs = []

    # while lap_time < 3:  # testing log
    while not done:
        steering, speed = controller.control(obs)  # each agent’s current observation
        print("steering = {}, speed = {}".format(round(steering, 5), speed))
        log_action.append([lap_time, steering, speed])

        obs, time_step, done, _ = env.step(np.array([[steering, speed]]))
        log_obs.append([lap_time, obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0]])

        lap_time += time_step
        env.render(mode='human')

    print('Sim elapsed time:', lap_time)
    xlsx_log_action(map_name, log_action)
    xlsx_log_observation(map_name, log_obs)


if __name__ == '__main__':
    main()
