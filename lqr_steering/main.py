"""
    MEAM 517 Final Project - LQR Steering Control - main application
    Author: Derek Zhou & Tancy Zhao
    References: https://github.com/f1tenth/f1tenth_gym/tree/main/examples
                https://github.com/f1tenth/f1tenth_planning/tree/main/f1tenth_planning/control/lqr
"""
import gym
import numpy as np
from lqr_steering import Waypoint, LQRSteeringController, Renderer


def main():
    # load waypoints into controller
    csv_file = 'Spielberg_raceline.csv'
    csv_data = np.loadtxt('./' + csv_file, delimiter=';', skiprows=0)
    waypoints = Waypoint(csv_data)

    controller = LQRSteeringController(waypoints)
    renderer = Renderer(waypoints)

    # create env & init
    map_name = 'Spielberg_map'
    env = gym.make('f110_gym:f110-v0', map='./' + map_name, map_ext='.png', num_agents=1)

    def render_callback(env_renderer):
        renderer.render_waypoints(env_renderer)  # render waypoints in env

    env.add_render_callback(render_callback)

    init_pos = np.array([[0.0, -0.84, 3.40]])
    obs, _, done, _ = env.reset(init_pos)

    # log data
    # log_steering = []
    # log_speed = []

    lap_time = 0.0
    while not done:
        steering, speed = controller.control(obs)  # each agent’s current observation
        print("steering = {}, speed = {}".format(steering, speed))
        # log_steering.append(steering)
        # log_speed.append(speed)
        obs, time_step, done, _ = env.step(np.array([[steering, speed]]))
        lap_time += time_step
        env.render(mode='human')

    print('Sim elapsed time:', lap_time)


if __name__ == '__main__':
    main()
