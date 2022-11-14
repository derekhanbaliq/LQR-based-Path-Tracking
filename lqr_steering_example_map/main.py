"""
    MEAM 517 Final Project - LQR Steering Control - main application
    Author: Derek Zhou & Tancy Zhao
    References: https://f1tenth-gym.readthedocs.io/en/latest/index.html
                https://github.com/f1tenth/f1tenth_gym/tree/main/examples
                https://github.com/f1tenth/f1tenth_planning/tree/main/f1tenth_planning/control/lqr
                https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathTracking/lqr_steer_control
"""

import gym
import numpy as np
from lqr_steering import Waypoint, LQRSteeringController, Renderer
from log import xlsx_log_action, xlsx_log_observation


def main():
    # load waypoints into controller
    csv_file = 'example_waypoints.csv'
    csv_data = np.loadtxt('./' + csv_file, delimiter=';', skiprows=0)
    waypoints = Waypoint(csv_data)

    controller = LQRSteeringController(waypoints)
    renderer = Renderer(waypoints)

    # create env & init
    map_name = 'example_map'
    env = gym.make('f110_gym:f110-v0', map='./' + map_name, map_ext='.png', num_agents=1)

    def render_callback(env_renderer):
        renderer.render_waypoints(env_renderer)  # render waypoints in env

    env.add_render_callback(render_callback)

    init_pos = np.array([[0.7, 0.0, 1.37079632679]])
    obs, _, done, _ = env.reset(init_pos)

    # placeholder for logging, plotting, and debugging
    log_action = []
    log_obs = []

    lap_time = 0.0

    # while lap_time < 0.03:  # testing log
    while not done:
        steering, speed = controller.control(obs)  # each agent’s current observation
        print("steering = {}, speed = {}".format(round(steering, 5), speed))
        log_action.append([lap_time, steering, speed])

        obs, time_step, done, _ = env.step(np.array([[steering, speed]]))
        log_obs.append([lap_time, obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0]])

        lap_time += time_step
        env.render(mode='human')

    print('Sim elapsed time:', lap_time)
    xlsx_log_action(log_action)
    xlsx_log_observation(log_obs)


if __name__ == '__main__':
    main()
