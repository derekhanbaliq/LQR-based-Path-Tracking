"""
    MEAM 517 Final Project - Stanley - main application
    Author: Derek Zhou & Tancy Zhao
    References: https://f1tenth-gym.readthedocs.io/en/latest/index.html
                https://github.com/f1tenth/f1tenth_gym/tree/main/examples
                https://github.com/f1tenth/f1tenth_planning/tree/main/f1tenth_planning/control/stanley
"""

import gym
import numpy as np
import yaml
import os

from log import xlsx_log_action, xlsx_log_observation
from stanley import StanleyPlanner, Waypoint
from render import Renderer
import math


def main():
    # load map & yaml
    map_name = 'example'  # Spielberg, example, MoscowRaceway, Catalunya
    map_path = os.path.abspath(os.path.join('..', 'map', map_name))
    yaml_config = yaml.load(open(map_path + '/' + map_name + '_map.yaml'), Loader=yaml.FullLoader)

    # load waypoints
    csv_data = np.loadtxt(map_path + '/' + map_name + '_raceline.csv', delimiter=';', skiprows=0)
    waypoints = None
    if map_name == 'Spielberg' or map_name == 'MoscowRaceway' or map_name == 'Catalunya':
        waypoints = np.vstack((csv_data[:, 1], csv_data[:, 2], csv_data[:, 5], csv_data[:, 3])).T
    elif map_name == 'example':
        waypoints = np.vstack((csv_data[:, 1], csv_data[:, 2], csv_data[:, 5], csv_data[:, 3] + math.pi/2)).T
    print(waypoints.shape)

    # load controller
    controller = StanleyPlanner(waypoints=waypoints)

    # create env & init
    env = gym.make('f110_gym:f110-v0', map=map_path + '/' + map_name + '_map', map_ext='.png', num_agents=1)
    render_wp = Waypoint(map_name, csv_data)
    renderer = Renderer(render_wp)
    env.add_render_callback(renderer.render_waypoints)

    # init
    init_pos = None
    if map_name == 'example':
        init_pos = np.array([[0.75, 0.0, 1.37079632679]])  # requires accurate init pos, otherwise crash
    else:
        init_pos = np.array([yaml_config['init_pos']])
    obs, _, done, _ = env.reset(init_pos)
    lap_time = 0.0

    # placeholder for logging, plotting, and debugging
    log_action = []
    log_obs = []

    # while lap_time < 3:  # testing log
    while not done:
        steering, speed = controller.plan(obs['poses_x'][0],
                                          obs['poses_y'][0],
                                          obs['poses_theta'][0],
                                          obs['linear_vels_x'][0],
                                          k_path=7.0)
        if speed >= 8.0:
            speed = 8.0  # speed limit < 8 m/s
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
