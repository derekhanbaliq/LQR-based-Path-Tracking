"""
    MEAM 517 Final Project - LQR Steering Control - LQR class
    Author: Derek Zhou & Tancy Zhao
    References: https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathTracking/lqr_steer_control
                https://github.com/f1tenth/f1tenth_planning/tree/main/f1tenth_planning/control/lqr
"""
import numpy as np
import math
from utils import calc_nearest_point, pi_2_pi
from pyglet.gl import GL_POINTS  # game interface


class Waypoint:

    def __init__(self, csv_data=None):
        self.x = csv_data[:, 0]
        self.y = csv_data[:, 1]
        self.v = csv_data[:, 2]
        self.θ = csv_data[:, 3]
        self.γ = csv_data[:, 4]


class CarState:

    def __init__(self, x=0.0, y=0.0, θ=0.0, v=0.0):
        self.x = x
        self.y = y
        self.θ = θ
        self.v = v


class LKVMState:
    """
    Linear Kinematic Vehicle Model's state space expression
    """

    def __init__(self, e_l=0.0, e_l_dot=0.0, e_θ=0.0, e_θ_dot=0.0):
        # 4 states
        self.e_l = e_l
        self.e_l_dot = e_l_dot
        self.e_θ = e_θ
        self.e_θ_dot = e_θ_dot
        # log old states
        self.old_e_l = 0.0
        self.old_e_θ = 0.0

    def update(self, e_l, e_θ, dt):
        self.e_l = e_l
        self.e_l_dot = (e_l - self.old_e_l) / dt
        self.e_θ = e_θ
        self.e_θ_dot = (e_θ - self.old_e_θ) / dt

        x = np.vstack([self.e_l, self.e_l_dot, self.e_θ, self.e_θ_dot])

        return x


class LQR:

    def __init__(self, dt, wheelbase, v=0.0):
        self.A = np.array([[1.0,     dt,        0,          0],
                           [0,       0,         v,          0],
                           [0,       0,         1,          dt],
                           [0,       0,         0,          0]])
        self.B = np.array([[0],
                           [0],
                           [0],
                           [v / wheelbase]])
        self.Q = np.diag([0.999, 0.0, 0.0066, 0.0])
        self.R = np.diag([0.75])

    def discrete_lqr(self):
        A = self.A
        B = self.B
        self.Q = np.diag([0.999, 0.0, 0.0066, 0.0])
        self.R = np.diag([0.75])
        Q = self.Q
        R = self.R

        M = np.zeros((Q.shape[0], R.shape[1]))
        MT = M.T

        S = self.solve_recatti_equation()
        K = np.linalg.pinv(B.T @ S @ B + R) @ (B.T @ S @ A + MT)  # u = -(B.T @ S @ B + R)^(-1) @ (B.T @ S @ A) @ x[k]

        return K  # K is 4 x 1

    def solve_recatti_equation(self):
        A = self.A
        B = self.B
        Q = self.Q
        R = self.R  # just for simplifying the following recatti expression

        S = self.Q
        Sn = None

        M = np.zeros((Q.shape[0], R.shape[1]))
        MT = M.T

        max_iter = 50
        ε = 0.001  # tolerance epsilon

        num_iteration = 0
        diff = math.inf  # without using iteration!
        tolerance = 0.001

        while num_iteration < max_iter and diff > tolerance:
            num_iteration += 1
        # for i in range(max_iter):
            Sn = Q + A.T @ S @ A - (A.T @ S @ B + M) @ np.linalg.pinv(R + B.T @ S @ B) @ (B.T @ S @ A + MT)
            if abs(Sn - S).max() < ε:
                break
            S = Sn

        return Sn


class LQRSteeringController:

    def __init__(self, waypoints):
        self.dt = 0.01  # time step
        self.wheelbase = 0.33
        self.waypoints = waypoints
        self.car = CarState()
        self.x = LKVMState()  # whenever create the controller, x exists - relatively static

    def control(self, curr_obs):
        """
            input car_state & waypoints
            output lqr-steering & pid-speed
        """
        self.car.x = curr_obs['poses_x'][0]
        self.car.y = curr_obs['poses_y'][0]
        self.car.θ = curr_obs['poses_theta'][0]
        self.car.v = curr_obs['linear_vels_x'][0]  # each agent’s current longitudinal velocity

        # input car_state, waypoints, timestep, matrix_q, matrix_r, iterations, eps)
        steering = self.lqr_steering_control()
        speed = self.pid_speed_control()

        return steering, speed

    def lqr_steering_control(self):
        """
        LQR steering control for Lateral Kinematics Vehicle Model - only steering for this part, consider feedforward
        """

        self.x.old_e_l = self.x.e_l
        self.x.old_e_θ = self.x.e_θ  # log into x's static variables

        e_l, e_θ, γ, v = self.calc_control_points()  # Calculate errors and reference point

        # print(e_l)
        # print(e_θ)
        # print(γ)
        # print(v)
        # print("---")

        lqr = LQR(self.dt, self.wheelbase, self.car.v)  # init A B Q R with the current car state
        K = lqr.discrete_lqr()  # use A, B, Q, R to get K
        print(K)

        x_new = self.x.update(e_l, e_θ, self.dt)  # x[k+1]
        # print(x_new)

        # feedback_term = pi_2_pi((-K @ x_new)[0, 0])  # K is 4 x 1 since u is 1 x 1, control steering only! - u_star
        feedback_term = (K @ x_new)[0, 0]
        # feedforward_term = math.atan2(self.wheelbase * γ, 1)  # = math.atan2(L / r, 1) = math.atan2(L, r)
        feedforward_term = self.wheelbase * γ

        steering = feedback_term + feedforward_term

        return steering

    def pid_speed_control(self):
        """
        TODO: full PID controller
        """
        front_pos = self.get_front_pos()
        _, _, _, i = calc_nearest_point(front_pos, np.array([self.waypoints.x, self.waypoints.y]).T)

        Kp = 1
        # speed = Kp * (self.waypoints.v[i] - self.car.v)  # desired speed - curr_spd
        # speed = 8.0  # just for debugging
        speed = self.waypoints.v[i]

        return speed

    def get_front_pos(self):
        front_x = self.car.x + self.wheelbase * math.cos(self.car.θ)
        front_y = self.car.y + self.wheelbase * math.sin(self.car.θ)
        front_pos = np.array([front_x, front_y])

        return front_pos

    def calc_control_points(self):
        front_pos = self.get_front_pos()

        waypoint_i, min_d, _, i = \
            calc_nearest_point(front_pos, np.array([self.waypoints.x, self.waypoints.y]).T)
        # print(np.array([self.waypoints.x, self.waypoints.y]).T)

        waypoint_to_front = front_pos - waypoint_i  # regard this as a vector

        front_axle_vec_rot_90 = np.array([[math.cos(self.car.θ - math.pi / 2.0)],
                                          [math.sin(self.car.θ - math.pi / 2.0)]])
        e_l = np.dot(waypoint_to_front.T, front_axle_vec_rot_90)  # real lateral error, the horizontal dist

        # NOTE: If your raceline is based on a different coordinate system you need to -+ pi/2 = 90 degrees
        e_θ = pi_2_pi(self.waypoints.θ[i] - self.car.θ)  # heading error
        γ = self.waypoints.γ[i]  # curvature of the nearst waypoint
        v = self.waypoints.v[i]  # velocity of the nearst waypoint

        return e_l, e_θ, γ, v


class Renderer:

    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.drawn_waypoints = []

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """

        # points = self.waypoints

        points = np.vstack((self.waypoints.x, self.waypoints.y)).T  # N x 2

        scaled_points = 50. * points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]
