"""
    MEAM 517 Final Project - LQR Steering Control - LQR class
    Author: Derek Zhou & Tancy Zhao
"""
import numpy as np
import scipy.linalg as la
import math


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


class LKVMParams:

    def __init__(self):
        self.dt = 0.01  # time step
        self.wheelbase = 0.33
        self.recatti_max_iter = 150
        self.recatti_ε = 0.01  # tolerance epsilon


class LKVMState:

    old_e_l = 0.0  # static variables for logging old errors
    old_e_θ = 0.0

    def __init__(self, e_l=0.0, e_l_dot=0.0, e_θ=0.0, e_θ_dot=0.0):
        # 4 states
        self.e_l = e_l
        self.e_l_dot = e_l_dot
        self.e_θ = e_θ
        self.e_θ_dot = e_θ_dot

    def update(self, e_l, e_θ, old_e_l, old_e_θ, dt):
        self.e_l = e_l
        self.e_l_dot = (e_l - old_e_l) / dt
        self.e_θ = e_θ
        self.e_θ_dot = (e_θ - old_e_θ) / dt

        return np.array([[self.e_l], [self.e_l_dot], [self.e_θ], [self.e_θ_dot]])


class LQR:

    def __init__(self, params, v=0):
        self.A = np.array([[1.0,     params.dt,  0,          0],
                           [0,       0,          v,          0],
                           [0,       0,          1,          params.dt],
                           [0,       0,          0,          0]])
        self.B = np.array([[0],
                           [0],
                           [0],
                           [v / params.wheelbase]])
        self.Q = np.eye(4)
        self.R = np.eye(1)

    def discrete_lqr(self, params):
        A = self.A
        B = self.B
        Q = self.Q
        R = self.R  # just for simplifying the following input expression

        S = self.solve_recatti_equation(params)
        K = la.inv(B.T @ S @ B + R) @ (B.T @ S @ A)  # u = -(B.T @ S @ B + R)^(-1) @ (B.T @ S @ A) @ x[k], K is 4 x 1

        return K

    def solve_recatti_equation(self, params):
        A = self.A
        B = self.B
        Q = self.Q
        R = self.R  # just for simplifying the following recatti expression

        S = self.Q
        Sn = None

        for i in range(params.recatti_max_iter):
            Sn = Q + A.T @ S @ A - (A.T @ S @ B) @ la.inv(R + B.T @ S @ B) @ (B.T @ S @ A)
            if abs(Sn - S).max() < params.recatti_ε:
                break
            S = Sn

        return Sn


class Controller:

    def __init__(self, waypoints):
        self.params = LKVMParams()
        self.waypoints = waypoints
        self.car_state = CarState()
        self.x = LKVMState()  # whenever create the controller, x exists - relatively static

    def control(self, curr_obs):
        """
            input car_state & waypoints
            output lqr-steering & pid-speed
        """
        self.car_state.x = curr_obs['poses_x'][0]
        self.car_state.y = curr_obs['poses_y'][0]
        self.car_state.θ = curr_obs['poses_theta'][0]
        self.car_state.v = curr_obs['linear_vels_x'][0]  # each agent’s current longitudinal velocity

        # input car_state, waypoints, timestep, matrix_q, matrix_r, iterations, eps)
        steering = lqr_steering_control(self.params, self.waypoints, self.car_state, self.x)
        speed = pid_speed_control(self.waypoints, self.car_state)

        return steering, speed


def lqr_steering_control(params, waypoints, car, x):

    lqr = LQR(params, car.v)  # init A, B, Q, R with the new car state

    i, e_l = calc_nearest_index(waypoints, car)
    # return nearst waypoint index & min dist with dir (e > 0, curr pos is on the left of the nearst waypoint)
    # nearst index & lateral error

    γ = waypoints.γ[i]  # curvature of nearst waypoint
    e_θ = pi_2_pi(car.θ - waypoints.θ[i])  # θ_e

    K = lqr.discrete_lqr(params)   # discrete LQR
    x_new = x.update(e_l, e_θ, x.old_e_l, x.old_e_θ, params.dt)
    feedback_term = pi_2_pi((-K @ x_new)[0, 0])  # K is 4 x 1 since u is 1 x 1, control steering only!

    # wheelbase * curvature = wheelbase / radius ?
    # = math.atan2(L / r, 1) = math.atan2(L, r) -> this can be drawn and understood easily
    # a compensation angle from feed-forward path
    feedforward_term = math.atan2(params.wheelbase * γ, 1)

    steering = feedback_term + feedforward_term

    x.old_e_l = e_l
    x.old_e_θ = e_θ

    return steering


def pid_speed_control(waypoints, car):
    i, _ = calc_nearest_index(waypoints, car)
    Kp = 1
    speed = Kp * (waypoints.v[i] - car.v)  # desired speed - curr_spd

    return speed


def calc_nearest_index(waypoints, car):
    d_x = [car.x - i for i in waypoints.x]  # [x, x, ... , x] - [cx_0, cx_1, ... , cx_n]
    d_y = [car.y - i for i in waypoints.y]
    d_sq = [idx ** 2 + idy ** 2 for (idx, idy) in zip(d_x, d_y)]  # zip(dx, dy) = [(dx0, dy0), (dx1, dy1) ...]

    min_d_sq = min(d_sq)
    index = d_sq.index(min_d_sq)  # index of min dist square
    min_d = math.sqrt(min_d_sq)  # min dist, not square of it

    # vector / point (dxl, dyl)
    min_d_x = waypoints.x[index] - car.x  # vector from x of current state to x of nearst point
    min_d_y = waypoints.y[index] - car.y

    # angle from x-axis to nearst point yaw dir - angle from x-axis to curr dir
    # limited in [-π/2, π/2]
    # it's not θ_e, only for detecting left or right
    angle = pi_2_pi(waypoints.θ[index] - math.atan2(min_d_y, min_d_x))
    if angle < 0:  # < 0, curr pos is on the right of the nearst way point dir
        min_d *= -1

    return index, min_d  # index & lateral error


def pi_2_pi(angle):

    return (angle + math.pi) % (2 * math.pi) - math.pi
