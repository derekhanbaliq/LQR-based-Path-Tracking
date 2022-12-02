import numpy
import time
import yaml
import gym
import sys
import numpy as np
from argparse import Namespace
from f110_gym.envs.dynamic_models import vehicle_dynamics_ks, vehicle_dynamics_st
from methods.pure_pursuit import *
from methods.path_follow_mpc import *
from models.kinematic import KinematicModel
from models.extended_kinematic import ExtendedKinematicModel
from helpers.closest_point import *
# import torch
# import gpytorch
# import os

from pyglet.gl import GL_POINTS
import json

from render import Renderer


@dataclass
class MPCConfig:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [accel, steering_speed]
    TK: int = 12  # finite time horizon length kinematic

    Rk: list = field(
        default_factory=lambda: np.diag([5.1, 100.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([1.1, 100.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 2.5, 13.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, velocity, heading]
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 2.5, 13.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, delta, v, yaw, yaw-rate, beta]
    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 3.0  # dist step [m] kinematic
    LENGTH: float = 4.298  # Length of the vehicle [m]
    WIDTH: float = 1.674  # Width of the vehicle [m]
    LR: float = 1.50876
    LF: float = 0.88392
    WB: float = 0.88392 + 1.50876  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
    MAX_SPEED: float = 45.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 11.5  # maximum acceleration [m/ss]
    MAX_DECEL: float = -45.0  # maximum acceleration [m/ss]

    MASS: float = 1225.887  # Vehicle mass


@dataclass
class MPCConfig_F110_6:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering angle, acceleration]
    TK: int = 8  # finite time horizon length, kinematic

    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 20.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_angle]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 20.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_angle]
    Qk: list = field(
        default_factory=lambda: np.diag([30, 30, 10, 10.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw, yaw-rate, beta]
    Qfk: list = field(
        default_factory=lambda: np.diag([30, 30, 10, 10.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, delta, v, yaw, yaw-rate, beta]
    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    # dlk: float = 0.03  # dist step [m] kinematic
    dlk: float = 0.2  # dist step [m] kinematic - ?????????????????????????????????? F1"TENTH"
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    LR: float = 0.17145
    LF: float = 0.15875
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad], 24.00°
    # MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s], 3.141592653589793
    MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]
    MAX_DECEL: float = -3.0  # maximum acceleration [m/ss]

    MASS: float = 3.74  # Vehicle mass


@dataclass
class MPCConfig_F110_8:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering angle, acceleration]
    TK: int = 8  # finite time horizon length, kinematic

    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 50.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_angle]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 50.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_angle]
    Qk: list = field(
        default_factory=lambda: np.diag([100, 100, 5.0, 10.0])
    )  # state error cost matrix, for the next (T) prediction time steps [x, y, v, yaw]
    Qfk: list = field(
        default_factory=lambda: np.diag([100, 100, 5.0, 10.0])
    )  # final state error matrix, penalty for the final state constraints: [x, y, v, yaw]
    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    # dlk: float = 0.03  # dist step [m] kinematic
    dlk: float = 0.2  # dist step [m] kinematic - check the difference between waypoints[0, 0] and waypoints[0, 1]
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    LR: float = 0.17145
    LF: float = 0.15875
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad], 24.00°
    # MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s], 3.141592653589793
    MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
    MAX_SPEED: float = 8.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]
    MAX_DECEL: float = -3.0  # maximum acceleration [m/ss]

    MASS: float = 3.74  # Vehicle mass


@dataclass
class MPCConfigEXT:
    NXK: int = 7  # length of kinematic state vector: z = [x, y, vx, yaw angle, vy, yaw rate, steering angle]
    NU: int = 2  # length of input vector: u = = [acceleration, steering speed]
    TK: int = 12  # finite time horizon length kinematic

    Rk: list = field(
        default_factory=lambda: np.diag([15.1, 100.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([10.1, 100.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 3.5, 13.0, 0.0, 0.0, 0.0])
    )  # state error cost matrix, for the next (T) prediction time steps
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 3.5, 13.0, 0.0, 0.0, 0.0])
    )  # final state error matrix, penalty  for the final state constraints
    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 3.0  # dist step [m] kinematic
    LENGTH: float = 4.298  # Length of the vehicle [m]
    WIDTH: float = 1.674  # Width of the vehicle [m]
    LR: float = 1.50876
    LF: float = 0.88392
    WB: float = 0.88392 + 1.50876  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
    MAX_SPEED: float = 45.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 11.5  # maximum acceleration [m/ss]
    MAX_DECEL: float = -45.0  # maximum acceleration [m/ss]

    MASS: float = 1225.887  # Vehicle mass


def draw_point(e, point, colour):
    scaled_point = 50. * point
    ret = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_point[0], scaled_point[1], 0]), ('c3B/stream', colour))
    return ret


class DrawDebug:
    def __init__(self):
        self.reference_traj_show = np.array([[0, 0]])
        self.predicted_traj_show = np.array([[0, 0]])
        self.dyn_obj_drawn = []
        self.f = 0

    def draw_debug(self, e):
        # delete dynamic objects
        while len(self.dyn_obj_drawn) > 0:
            if self.dyn_obj_drawn[0] is not None:
                self.dyn_obj_drawn[0].delete()
            self.dyn_obj_drawn.pop(0)

        # spawn new objects
        for p in self.reference_traj_show:
            self.dyn_obj_drawn.append(draw_point(e, p, [255, 0, 0]))

        for p in self.predicted_traj_show:
            self.dyn_obj_drawn.append(draw_point(e, p, [0, 255, 0]))


def main():
    """
    main entry point
    """

    # Choose model to use
    model_to_use = 'kinematic'  # options: kinematic, ext_kinematic

    # # load friction map
    # tpamap_name = './maps/rounded_rectangle/rounded_rectangle_tpamap.csv'
    # tpadata_name = './maps/rounded_rectangle/rounded_rectangle_tpadata.json'
    #
    # tpamap = np.loadtxt(tpamap_name, delimiter=';', skiprows=1)
    # tpamap *= 1.5  # map is 1.5 times larger than normal
    #
    # tpadata = {}
    # with open(tpadata_name) as f:
    #     tpadata = json.load(f)

    # Creating the single-track Motion planner and Controller

    # Init regulator
    # work = {'mass': 1225.88, 'lf': 0.80597534362552312, 'tlad': 2.6461887897713965, 'vgain': 0.950338203837889}

    # Load config file
    with open('config_MoscowRaceway_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # Load waypoints
    # raceline = np.loadtxt('./maps/rounded_rectangle/rounded_rectangle_waypoints.csv', delimiter=";", skiprows=3)
    raceline = np.loadtxt('./maps/MoscowRaceway/MoscowRaceway_raceline.csv', delimiter=";", skiprows=3)
    waypoints = np.array(raceline)
    # waypoints[:, 5] = waypoints[:, 5] / 2  # max 4 m/s

    # Choose planner
    planner_st_mpc = None
    if model_to_use == 'kinematic':
        planner_st_mpc = STMPCPlanner(model=KinematicModel(config=MPCConfig_F110_6()), waypoints=waypoints, config=MPCConfig_F110_6())
    elif model_to_use == 'ext_kinematic':
        planner_st_mpc = STMPCPlanner(model=ExtendedKinematicModel(config=MPCConfigEXT()), waypoints=waypoints,
                                      config=MPCConfigEXT())
    else:
        print("ERROR: Unknown vehicle model")
        exit(1)

    # planner_pp = PurePursuitPlanner(conf, 0.17145 + 0.15875)

    draw = DrawDebug()

    def render_callback(env_renderer):
        # custom extra drawing function
        e = env_renderer
    #
    #     # update camera to follow car
    #     x = e.cars[0].vertices[::2]
    #     y = e.cars[0].vertices[1::2]
    #     top, bottom, left, right = max(y), min(y), min(x), max(x)
    #     e.score_label.x = left
    #     e.score_label.y = top - 400
    #     e.left = left - 500
    #     e.right = right + 500
    #     e.top = top + 500
    #     e.bottom = bottom - 500
    #
    #     # planner_pp.render_waypoints(e)
        draw.draw_debug(e)

    # MB - reference point: center of mass
    # ST - reference point: center of mass
    env = None
    if model_to_use == 'kinematic':
        env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                       num_agents=1, timestep=0.001, model='MB', drive_control_mode='acc',
                       steering_control_mode='angle')
    elif model_to_use == 'ext_kinematic':
        env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                       num_agents=1, timestep=0.001, model='MB', drive_control_mode='acc',
                       steering_control_mode='vel')

    env.add_render_callback(render_callback)
    # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
    # obs, step_reward, done, info = env.reset(np.array([[conf.sx * 10, conf.sy * 10, conf.stheta, 0.0, 10.0, 0.0, 0.0]]))
    obs, _, done, _ = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    # derek's waypoints renderer
    renderer = Renderer(waypoints)
    env.add_render_callback(renderer.render_waypoints)

    laptime = 0.0
    start = time.time()
    render_every = 2
    last_render = 0

    log = {}
    log['time'] = []

    control_step = 20.0  # ms
    num_of_sim_steps = int(control_step / (env.timestep * 1000.0))  # 20

    while not done:
        # # Regulator step pure pursuit
        # speed2, steer_angle = planner_pp.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'],
        #                                       work['vgain'])

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

        u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = planner_st_mpc.plan(vehicle_state)

        if model_to_use == 'kinematic':
            pass
        elif model_to_use == 'ext_kinematic':
            u[0] = u[0] * planner_st_mpc.config.MASS  # output of the kinematic MPC is not acceleration, but force

        # draw predicted states and reference trajectory
        draw.reference_traj_show = np.array([mpc_ref_path_x, mpc_ref_path_y]).T
        draw.predicted_traj_show = np.array([mpc_pred_x, mpc_pred_y]).T

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

        # Logging
        log['time'].append(laptime)

        if obs['lap_counts'][0] == 1:
            done = 1

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
    with open('log01', 'w') as f:
        json.dump(log, f)


if __name__ == '__main__':
    main()

# formula zero paper
# exp2
