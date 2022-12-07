"""
    Single Track Kinematic MPC - kinematic model
    Author: Hongrui Zheng, Johannes Betz, Ahmad Amine, Derek Zhou
    References: https://github.com/f1tenth/f1tenth_planning/tree/main/f1tenth_planning/control/kinematic_mpc
                https://github.com/f1tenth/f1tenth_planning/tree/main/examples/control
                https://atsushisakai.github.io/PythonRobotics/modules/path_tracking/model_predictive_speed_and_steering_control/model_predictive_speed_and_steering_control.html#mpc-modeling
                https://www.cvxpy.org/
                https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathTracking/model_predictive_speed_and_steer_control
"""
import numpy as np


class KinematicModel:
    """
    states - [x, y, v, yaw]
    inputs - [acceleration, steering angle]
    reference point - center of rear axle
    """
    def __init__(self, config):
        self.config = config

    def clip_input(self, u):
        # u matrix N x 2
        u = np.clip(u, [self.config.MAX_DECEL, self.config.MIN_STEER], [self.config.MAX_ACCEL, self.config.MAX_STEER])
        # numpy.clip(a, a_min, a_max, out=None, **kwargs), Clip (limit) the values in an array.

        return u

    def clip_output(self, state):
        # state matrix N x 4
        state[2] = np.clip(state[2], self.config.MIN_SPEED, self.config.MAX_SPEED)  # speed only

        return state

    def get_model_constraints(self):
        state_constraints = np.array([[-np.inf, -np.inf, self.config.MIN_SPEED, -np.inf],
                                      [np.inf, np.inf, self.config.MAX_SPEED, np.inf]])

        input_constraints = np.array([[self.config.MAX_DECEL, self.config.MIN_STEER],
                                      [self.config.MAX_ACCEL, self.config.MAX_STEER]])
        input_diff_constraints = np.array([[-np.inf, -self.config.MAX_STEER_V * self.config.DTK],
                                           [np.inf, self.config.MAX_STEER_V * self.config.DTK]])

        return state_constraints, input_constraints, input_diff_constraints

    def sort_reference_trajectory(self, position_ref, yaw_ref, speed_ref):
        reference = np.array([position_ref[:, 0], position_ref[:, 1], speed_ref, yaw_ref])  # x, y, v, yaw

        return reference  # N x 4

    def get_general_states(self, state):
        speed = state[2]
        orientation = state[3]
        position = state[[0, 1]]

        return speed, orientation, position  # express the states more generally

    def get_f(self, state, control_input):
        # state = x, y, v, yaw
        clipped_control_input = self.clip_input(control_input)  # input check
        delta = clipped_control_input[1]
        a = clipped_control_input[0]

        # f is for Forward Euler Discretization with sampling time dt: z[k+1] = z[k] + f(z[k], u[k]) * dt
        f = np.zeros(4)
        f[0] = state[2] * np.cos(state[3])  # x_dot
        f[1] = state[2] * np.sin(state[3])  # y_dot
        f[3] = state[2] / self.config.WB * np.tan(delta)  # yaw_dot
        f[2] = a  # v_dot

        return f  # kinematic model f(x[k], u[k]), Automatic Steering P27 or Atsushi's KMPC doc

    def get_model_matrix(self, state, u):
        """
        https://atsushisakai.github.io/PythonRobotics/modules/path_tracking/model_predictive_speed_and_steering_control/model_predictive_speed_and_steering_control.html#mpc-modeling
        Calculate kinematic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax + Bu + C
        State vector: x=[x, y, v, yaw]
        """
        v = state[2]
        phi = state[3]
        delta = u[1]

        # State (or system) matrix A, 4 x 4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * np.cos(phi)
        A[0, 3] = -self.config.DTK * v * np.sin(phi)
        A[1, 2] = self.config.DTK * np.sin(phi)
        A[1, 3] = self.config.DTK * v * np.cos(phi)
        A[3, 2] = self.config.DTK * np.tan(delta) / self.config.WB

        # Input Matrix B, 4 x 2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * np.cos(delta) ** 2)

        # Matrix C, 4 x 1, C is just a shift because we need an affine model
        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * np.sin(phi) * phi
        C[1] = -self.config.DTK * v * np.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * np.cos(delta) ** 2)

        return A, B, C

    def predict_motion(self, x0, control_input):
        predicted_states = np.zeros((self.config.NXK, self.config.TK + 1))  # 4 x (8 + 1)
        predicted_states[:, 0] = x0  # set current state
        state = x0
        for i in range(1, self.config.TK + 1):  # 1 ... 8
            # Forward Euler Discretization with sampling time dt: z[k+1] = z[k] + f(z[k], u[k]) * dt
            state = state + self.get_f(state, control_input[:, i - 1]) * self.config.DTK
            state = self.clip_output(state)
            predicted_states[:, i] = state

        input_prediction = np.zeros((self.config.NU, self.config.TK + 1))  # 2 x (8 + 1), empty!

        return predicted_states, input_prediction  # filled states, empty inputs
