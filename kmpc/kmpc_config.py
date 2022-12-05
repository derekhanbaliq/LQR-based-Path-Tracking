"""
    Single Track Kinematic MPC - KMPC config parameters
    Author: Hongrui Zheng, Johannes Betz, Ahmad Amine, Derek Zhou
    References: https://github.com/f1tenth/f1tenth_planning/tree/main/f1tenth_planning/control/kinematic_mpc
                https://github.com/f1tenth/f1tenth_planning/tree/main/examples/control
                https://atsushisakai.github.io/PythonRobotics/modules/path_tracking/model_predictive_speed_and_steering_control/model_predictive_speed_and_steering_control.html#mpc-modeling
                https://www.cvxpy.org/
                https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathTracking/model_predictive_speed_and_steer_control

"""
from dataclasses import dataclass, field
from kmpc import *

"""
Derek's comment:
    parameters of KMPC is hard to tune, because all the parameters are dependent.
    High-speed condition is challenging for KMPC, since it has taken forces and friction into account
"""


@dataclass
class MPCConfig_F110_6:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering angle, acceleration]
    TK: int = 8  # finite time horizon length, kinematic

    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 25.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_angle]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 25.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_angle]
    Qk: list = field(
        default_factory=lambda: np.diag([25, 25, 10, 10])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v]
    Qfk: list = field(
        default_factory=lambda: np.diag([25, 25, 10, 10])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, delta, v]
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
    NXK: int = 4
    NU: int = 2
    TK: int = 10  # rising to 12 will cut corners and unstable straight running

    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 35.0])  # 25 will cause unstable straight running
    )  # input cost matrix, penalty for inputs - [accel, steering_angle]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 35.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_angle]
    Qk: list = field(
        default_factory=lambda: np.diag([25, 25, 5.0, 20])
    )  # state error cost matrix, for the next (T) prediction time steps [x, y, v, yaw]
    Qfk: list = field(
        default_factory=lambda: np.diag([25, 25, 5.0, 20])
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
