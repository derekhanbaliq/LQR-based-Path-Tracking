"""
    Single Track Kinematic MPC - main
    Author: Hongrui Zheng, Johannes Betz, Ahmad Amine, Derek Zhou
    References: https://github.com/f1tenth/f1tenth_planning/tree/main/f1tenth_planning/control/kinematic_mpc
                https://github.com/f1tenth/f1tenth_planning/tree/main/examples/control
                https://atsushisakai.github.io/PythonRobotics/modules/path_tracking/model_predictive_speed_and_steering_control/model_predictive_speed_and_steering_control.html#mpc-modeling
                https://www.cvxpy.org/
                https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathTracking/model_predictive_speed_and_steer_control
"""
import gym
import os
import yaml
import time
from kinematic_model import KinematicModel
from extended_kinematic_model import ExtendedKinematicModel
from kmpc import *
from kmpc_config import MPCConfig_F110_6, MPCConfig_F110_8, MPCConfigEXT
from render import Renderer


def main():
    method_name = 'kmpc'
    model_name = 'kinematic'  # options: kinematic, ext_kinematic
    model_config = MPCConfig_F110_8()

    # load map & yaml
    map_name = 'example'  # example, Catalunya, MoscowRaceway - 6, Spielberg collided
    map_path = os.path.abspath(os.path.join('..', 'maps', map_name))
    yaml_config = yaml.load(open(map_path + '/' + map_name + '_map.yaml'), Loader=yaml.FullLoader)

    # load waypoints
    raceline = np.loadtxt(map_path + '/' + map_name + '_raceline.csv', delimiter=';', skiprows=0)
    waypoints = np.array(raceline)

    # load controller
    controller = None
    if model_name == 'kinematic':
        controller = KMPCController(model=KinematicModel(config=model_config), waypoints=waypoints, config=model_config)
    elif model_name == 'ext_kinematic':
        controller = KMPCController(model=ExtendedKinematicModel(config=MPCConfigEXT()), waypoints=waypoints,
                                    config=MPCConfigEXT())
    else:
        print("ERROR: Unknown vehicle model")
        exit(1)

    # create env & init
    env = gym.make('f110_gym:f110-v0', map=map_path + '/' + map_name + '_map', map_ext='.png', num_agents=1)
    init_pos = np.array([yaml_config['init_pos']])
    obs, _, done, _ = env.reset(init_pos)

    control_step = 20.0  # ms
    num_of_sim_steps = int(control_step / (env.timestep * 1000.0))  # 20

    # render init
    renderer = Renderer(waypoints)
    env.add_render_callback(renderer.render_waypoints)
    env.add_render_callback(renderer.draw_debug)
    render_every = 2  # render every 2 steps
    last_render = 0

    laptime = 0.0
    start = time.time()

    while not done:
        # Regulator step MPC
        vehicle_state = None
        if model_name == 'kinematic':
            vehicle_state = np.array([env.sim.agents[0].state[0],
                                      env.sim.agents[0].state[1],
                                      env.sim.agents[0].state[3],  # vx
                                      env.sim.agents[0].state[4],  # yaw angle
                                      ])
        elif model_name == 'ext_kinematic':
            vehicle_state = np.array([env.sim.agents[0].state[0],
                                      env.sim.agents[0].state[1],
                                      env.sim.agents[0].state[3],  # vx
                                      env.sim.agents[0].state[4],  # yaw angle
                                      env.sim.agents[0].state[10],  # vy
                                      env.sim.agents[0].state[5],  # yaw rate
                                      env.sim.agents[0].state[2],  # steering angle
                                      ])

        steering, speed, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = \
            controller.control(vehicle_state)

        if model_name == 'ext_kinematic':
            speed = speed * controller.config.MASS  # output of the kinematic MPC is not acceleration, but force
            # TODO: used to be accel, need to be modified!

        # draw predicted states and reference trajectory
        renderer.reference_traj_show = np.array([mpc_ref_path_x, mpc_ref_path_y]).T
        renderer.predicted_traj_show = np.array([mpc_pred_x, mpc_pred_y]).T

        # update simulation step & time
        step_reward = 0.0
        for i in range(num_of_sim_steps):  # 20
            obs, rew, done, info = env.step(np.array([[steering, speed]]))
            print("steering = {}, speed = {}".format(round(steering, 5), round(speed, 5)))
            step_reward += rew
        laptime += step_reward  # update timestamp

        # update rendering
        last_render += 1
        if last_render == render_every:  # render every 2 steps
            last_render = 0
        env.render(mode='human_fast')

        if obs['lap_counts'][0] == 1:
            done = 1

    print('Sim elapsed time:', round(laptime, 5), 'Real elapsed time:', round(time.time() - start, 5))


if __name__ == '__main__':
    main()
