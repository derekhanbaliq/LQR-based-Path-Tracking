import gym
import os
import yaml
import time
from kinematic_model import KinematicModel
from extended_kinematic_model import ExtendedKinematicModel
from kmpc import *
from closest_point import *

from render import Renderer
from kmpc_config import MPCConfig, MPCConfig_F110_6, MPCConfig_F110_8, MPCConfigEXT


def main():
    method_name = 'kmpc'
    model_to_use = 'kinematic'  # options: kinematic, ext_kinematic

    # load map & yaml
    map_name = 'MoscowRaceway'  # MoscowRaceway, stadium
    map_path = os.path.abspath(os.path.join('..', 'map', map_name))
    yaml_config = yaml.load(open(map_path + '/' + map_name + '_map.yaml'), Loader=yaml.FullLoader)

    # load waypoints
    raceline = np.loadtxt(map_path + '/' + map_name + '_raceline.csv', delimiter=';', skiprows=0)
    waypoints = np.array(raceline)

    # load friction map
    # tpamap_name = './maps/rounded_rectangle/rounded_rectangle_tpamap.csv'
    # tpadata_name = './maps/rounded_rectangle/rounded_rectangle_tpadata.json'
    # tpamap = np.loadtxt(tpamap_name, delimiter=';', skiprows=1)
    # tpamap *= 1.5  # map is 1.5 times larger than normal
    # tpadata = {}
    # with open(tpadata_name) as f:
    #     tpadata = json.load(f)

    # load controller
    controller = None
    if model_to_use == 'kinematic':
        controller = STMPCPlanner(model=KinematicModel(config=MPCConfig_F110_6()), waypoints=waypoints,
                                         config=MPCConfig_F110_6())
    elif model_to_use == 'ext_kinematic':
        controller = STMPCPlanner(model=ExtendedKinematicModel(config=MPCConfigEXT()), waypoints=waypoints,
                                         config=MPCConfigEXT())
    else:
        print("ERROR: Unknown vehicle model")
        exit(1)

    # create env & init
    env = gym.make('f110_gym:f110-v0', map=map_path + '/' + map_name + '_map', map_ext='.png', num_agents=1)
    # MB - reference point: center of mass
    # ST - reference point: center of mass
    # env = None
    # if model_to_use == 'kinematic':
    #     env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
    #                    num_agents=1, timestep=0.001, model='MB', drive_control_mode='acc',
    #                    steering_control_mode='angle')
    # elif model_to_use == 'ext_kinematic':
    #     env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
    #                    num_agents=1, timestep=0.001, model='MB', drive_control_mode='acc',
    #                    steering_control_mode='vel')

    # waypoints renderer
    renderer = Renderer(waypoints)
    env.add_render_callback(renderer.render_waypoints)
    env.add_render_callback(renderer.draw_debug)

    init_pos = np.array([yaml_config['init_pos']])
    obs, _, done, _ = env.reset(init_pos)

    laptime = 0.0
    start = time.time()
    render_every = 2
    last_render = 0

    control_step = 20.0  # ms
    num_of_sim_steps = int(control_step / (env.timestep * 1000.0))  # 20

    while not done:
        # Regulator step MPC
        vehicle_state = None
        if model_to_use == 'kinematic':
            vehicle_state = np.array([env.sim.agents[0].state[0],
                                      env.sim.agents[0].state[1],
                                      env.sim.agents[0].state[3],  # vx
                                      env.sim.agents[0].state[4],  # yaw angle
                                      ])
        elif model_to_use == 'ext_kinematic':
            vehicle_state = np.array([env.sim.agents[0].state[0],
                                      env.sim.agents[0].state[1],
                                      env.sim.agents[0].state[3],  # vx
                                      env.sim.agents[0].state[4],  # yaw angle
                                      env.sim.agents[0].state[10],  # vy
                                      env.sim.agents[0].state[5],  # yaw rate
                                      env.sim.agents[0].state[2],  # steering angle
                                      ])

        u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = controller.plan(vehicle_state)

        if model_to_use == 'kinematic':
            pass
        elif model_to_use == 'ext_kinematic':
            u[0] = u[0] * controller.config.MASS  # output of the kinematic MPC is not acceleration, but force

        # draw predicted states and reference trajectory
        renderer.reference_traj_show = np.array([mpc_ref_path_x, mpc_ref_path_y]).T
        renderer.predicted_traj_show = np.array([mpc_pred_x, mpc_pred_y]).T

        # # set correct friction to the environment
        # min_id = get_closest_point_vectorized(np.array([obs['poses_x'][0], obs['poses_y'][0]]), np.array(tpamap))
        # env.params['tire_p_dy1'] = 1.0  # tpadata[str(min_id)][0]  # mu_y
        # env.params['tire_p_dx1'] = 1.1  # tpadata[str(min_id)][0] * 1.1  # mu_x

        # Simulation step
        step_reward = 0.0
        for i in range(num_of_sim_steps):  # 20
            obs, rew, done, info = env.step(np.array([[u[1], u[0]]]))
            print("steering = {}, speed = {}".format(round(u[1], 5), u[0]))
            step_reward += rew
        laptime += step_reward

        # Rendering
        last_render += 1
        if last_render == render_every:
            last_render = 0
        env.render(mode='human_fast')

        if obs['lap_counts'][0] == 1:
            done = 1

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)


if __name__ == '__main__':
    main()
