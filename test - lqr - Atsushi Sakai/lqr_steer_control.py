"""

Path tracking simulation with LQR steering control and PID speed control.

author Atsushi Sakai (@Atsushi_twi)

"""
import scipy.linalg as la
import matplotlib.pyplot as plt
import math
import numpy as np
import sys
import pathlib

# sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import cubic_spline_planner

Kp = 1.0  # speed proportional gain

# LQR parameter
Q = np.eye(4)
R = np.eye(1)

# parameters
dt = 0.1  # time tick[s]
L = 0.5  # Wheelbase of the vehicle [m]
max_steer = np.deg2rad(45.0)  # maximum steering angle[rad]

show_animation = True


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


def update(state, a, delta):
    if delta >= max_steer:
        delta = max_steer
    if delta <= - max_steer:
        delta = - max_steer

    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.v = state.v + a * dt

    return state


def PIDControl(target, current):
    a = Kp * (target - current)

    return a


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def solve_DARE(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 150
    eps = 0.01  # tolerance epsilon
    Xn = None

    for i in range(maxiter):
        Xn = Q + A.T @ X @ A - A.T @ X @ B @ la.inv(R + B.T @ X @ B) @ B.T @ X @ A
        if (abs(Xn - X)).max() < eps:
            break
        X = Xn

    return Xn


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_DARE(A, B, Q, R)

    # compute the LQR gain
    K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)  # u = -(B.T @ S @ B + R)^(-1) @ (B.T @ S @ A) @ x[k], K is 4 x 1

    eigVals, eigVecs = la.eig(A - B @ K)  # lambda_cl from A_cl

    return K, X, eigVals


def lqr_steering_control(state, cx, cy, cyaw, ck, pe, pth_e):
    ind, e = calc_nearest_index(state, cx, cy, cyaw)
    # return nearst waypoint index & min dist with dir (e > 0, curr pos is on the left of the nearst waypoint)
    # nearst index & lateral error

    k = ck[ind]  # curvature of nearst waypoint
    v = state.v  # state velocity
    th_e = pi_2_pi(state.yaw - cyaw[ind])  # θ_e

    A = np.zeros((4, 4))
    A[0, 0] = 1.0
    A[0, 1] = dt
    A[1, 2] = v
    A[2, 2] = 1.0
    A[2, 3] = dt
    # print(A)

    B = np.zeros((4, 1))
    B[3, 0] = v / L

    K, _, _ = dlqr(A, B, Q, R)  # discrete LQR

    x = np.zeros((4, 1))

    x[0, 0] = e
    x[1, 0] = (e - pe) / dt
    x[2, 0] = th_e
    x[3, 0] = (th_e - pth_e) / dt

    # wheelbase * curvature = wheelbase / radius ?
    # = math.atan2(L / r, 1) = math.atan2(L, r) -> this can be drawn and understood easily
    # a compensation angle from feed-forward path
    ff = math.atan2(L * k, 1)
    fb = pi_2_pi((-K @ x)[0, 0])  # K is 4 x 1 since u is 1 x 1, control steering only!
    print("ff = {}".format(ff))
    print("fb = {}".format(fb))

    delta = ff + fb

    return delta, ind, e, th_e


def calc_nearest_index(state, cx, cy, cyaw):
    dx = [state.x - icx for icx in cx]  # [x, x, ... , x] - [cx_0, cx_1, ... , cx_n]
    dy = [state.y - icy for icy in cy]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]  # zip(dx, dy) = [(dx0, dy0), (dx1, dy1) ...]

    mind = min(d)

    ind = d.index(mind)  # index of min dist

    mind = math.sqrt(mind)  # min dist, not square of it

    # vector / point (dxl, dyl)
    dxl = cx[ind] - state.x  # vector from x of current state to x of nearst point
    dyl = cy[ind] - state.y

    # angle from x-axis to nearst point yaw dir - angle from x-axis to curr dir
    # limited in [-π/2, π/2]
    # it's not θ_e
    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:  # < 0, curr pos is on the right of the nearst way point dir
        mind *= -1

    return ind, mind  # index & lateral error


def closed_loop_prediction(cx, cy, cyaw, ck, speed_profile, goal):
    T = 500.0  # max simulation time
    goal_dis = 0.3  # goal range in 0.3
    stop_speed = 0.05

    state = State(x=-0.0, y=-0.0, yaw=0.0, v=0.0)  # declare a "state" object

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]  # init
    t = [0.0]

    e, e_th = 0.0, 0.0

    while T >= time:
        # dl? target index, cross-tracking error, heading error
        dl, target_ind, e, e_th = lqr_steering_control(state, cx, cy, cyaw, ck, e, e_th)

        ai = PIDControl(speed_profile[target_ind], state.v)  # speed profile (sp) is the desired speed of nearst point
        state = update(state, ai, dl)

        if abs(state.v) <= stop_speed:
            target_ind += 1  # the step is so close, pick the next one waypoint

        time = time + dt  # update time + 0.1s

        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        # math.hypot() returns the square root of the sum of squares of its arguments.
        if math.hypot(dx, dy) <= goal_dis:  # the euclidean dist between goal and curr pos <= 0.3
            print("Goal")
            break  # jump out while loop

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        if target_ind % 1 == 0 and show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("speed[km/h]:" + str(round(state.v * 3.6, 2))
                      + ",target index:" + str(target_ind))
            plt.pause(0.0001)

    return t, x, y, yaw, v


def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)  # list = [spd, spd, ... , spd]

    direction = 1.0

    # Set stop point
    for i in range(len(cx) - 1):
        dyaw = abs(cyaw[i + 1] - cyaw[i])
        switch = math.pi / 4.0 <= dyaw < math.pi / 2.0  # check if π/4<= dyaw <= π/2, switch = True / False

        if switch:
            direction *= -1

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

        if switch:
            speed_profile[i] = 0.0

    speed_profile[-1] = 0.0  # stop

    return speed_profile


def main():
    print("LQR steering control tracking start!!")
    ax = [0.0, 6.0, 12.5, 10.0, 7.5, 3.0, -1.0]
    ay = [0.0, -3.0, -5.0, 6.5, 3.0, 5.0, -2.0]
    goal = [ax[-1], ay[-1]]

    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)
    target_speed = 10.0 / 3.6  # simulation parameter km/h -> m/s
    # print("len(cx) = {}".format(len(cx)))  # = 426
    # ax: x coordinates of existing points
    # ay: y coordinates of existing points
    # ds: interpolation step
    # cx: x coordinates of interpolated points
    # cy: y coordinates of interpolated points
    # cyaw: dirs of interpolated points
    # ck: curvatures of interpolated points
    # s: interpolated point lists

    sp = calc_speed_profile(cx, cy, cyaw, target_speed)  # modified speeds for each interpolated points
    # print("sp = {}".format(sp))  # all 2.77, last one is zero

    t, x, y, yaw, v = closed_loop_prediction(cx, cy, cyaw, ck, sp, goal)

    # if show_animation:  # pragma: no cover
    #     plt.close()
    #     plt.subplots(1)
    #     plt.plot(ax, ay, "xb", label="input")
    #     plt.plot(cx, cy, "-r", label="spline")
    #     plt.plot(x, y, "-g", label="tracking")
    #     plt.grid(True)
    #     plt.axis("equal")
    #     plt.xlabel("x[m]")
    #     plt.ylabel("y[m]")
    #     plt.legend()
    #
    #     plt.subplots(1)
    #     plt.plot(s, [np.rad2deg(iyaw) for iyaw in cyaw], "-r", label="yaw")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.xlabel("line length[m]")
    #     plt.ylabel("yaw angle[deg]")
    #
    #     plt.subplots(1)
    #     plt.plot(s, ck, "-r", label="curvature")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.xlabel("line length[m]")
    #     plt.ylabel("curvature [1/m]")
    #
    #     plt.show()


if __name__ == '__main__':
    main()
