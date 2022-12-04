"""
    Single Track Kinematic MPC waypoint tracker
    Author: Hongrui Zheng, Johannes Betz, Ahmad Amine, Derek Zhou
    References: https://github.com/f1tenth/f1tenth_planning/tree/main/f1tenth_planning/control/kinematic_mpc
                https://github.com/f1tenth/f1tenth_planning/tree/main/examples/control
                https://atsushisakai.github.io/PythonRobotics/modules/path_tracking/model_predictive_speed_and_steering_control/model_predictive_speed_and_steering_control.html#mpc-modeling
                https://www.cvxpy.org/
                https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathTracking/model_predictive_speed_and_steer_control

"""
import cvxpy
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from numba import njit
import math


@njit(cache=True)
def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the env
    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points
                   on the trajectory. (p_i---*-------p_i+1)
        i (int): index of the nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    dist_from_segment_start = np.linalg.norm(diffs[min_dist_segment] * t[min_dist_segment])

    return projections[min_dist_segment], dist_from_segment_start, dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@njit(cache=True)
def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class KMPCController:
    """
    Single Track Kinematic MPC Controller
    waypoints are just whole CSV data
    """

    def __init__(self, model, config, waypoints=None):
        self.waypoints = waypoints
        self.model = model
        self.config = config
        self.waypoints_distances = np.linalg.norm(self.waypoints[1:, (1, 2)] - self.waypoints[:-1, (1, 2)], axis=1)
        self.target_ind = 0
        self.input_o = np.zeros(self.config.NU) * np.NAN
        self.odelta_v = np.NAN
        self.oa = np.NAN
        self.origin_switch = 1
        self.init_flag = 0
        self.mpc_prob_init()

    def control(self, states, waypoints=None):
        """
        input waypoints and current car states, execute MPC to return steering and speed with other data for logging
        """
        if waypoints is not None:
            if waypoints.shape[1] < 3 or len(waypoints.shape) != 2:
                raise ValueError("Waypoints needs to be a (Nxm), m >= 3, numpy array!")
            self.waypoints = waypoints
            self.waypoints_distances = np.linalg.norm(self.waypoints[1:, (1, 2)] - self.waypoints[:-1, (1, 2)], axis=1)
        else:
            if self.waypoints is None:
                raise ValueError("Please set waypoints to track during planner instantiation or when calling plan()")

        steering, speed, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = \
            self.MPC_Control(states, self.waypoints)

        return steering, speed, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy

    def get_reference_trajectory(self, predicted_speeds, dist_from_segment_start, idx, waypoints):
        s_relative = np.zeros((self.config.TK + 1,))
        s_relative[0] = dist_from_segment_start
        s_relative[1:] = predicted_speeds * self.config.DTK
        s_relative = np.cumsum(s_relative)

        waypoints_distances_relative = np.cumsum(np.roll(self.waypoints_distances, -idx))

        index_relative = np.int_(np.ones((self.config.TK + 1,)))
        for i in range(self.config.TK + 1):
            index_relative[i] = (waypoints_distances_relative <= s_relative[i]).sum()
        index_absolute = np.mod(idx + index_relative, waypoints.shape[0] - 1)

        segment_part = s_relative - (
                waypoints_distances_relative[index_relative] - self.waypoints_distances[index_absolute])

        t = (segment_part / self.waypoints_distances[index_absolute])
        # print(np.all(np.logical_and((t < 1.0), (t > 0.0))))

        position_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, (1, 2)] -
                          waypoints[index_absolute][:, (1, 2)])

        orientation_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 3] -
                             waypoints[index_absolute][:, 3])

        speed_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 5] -
                       waypoints[index_absolute][:, 5])

        interpolated_positions = waypoints[index_absolute][:, (1, 2)] + (t * position_diffs.T).T

        interpolated_orientations = waypoints[index_absolute][:, 3] + (t * orientation_diffs)
        interpolated_orientations = (interpolated_orientations + np.pi) % (2 * np.pi) - np.pi

        interpolated_speeds = waypoints[index_absolute][:, 5] + (t * speed_diffs)

        # rearrange reference data
        reference = self.model.sort_reference_trajectory(interpolated_positions,
                                                         interpolated_orientations,
                                                         interpolated_speeds)

        return reference

    def calc_ref_trajectory(self, position, orientation, speed, path):
        """
        https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathTracking/model_predictive_speed_and_steer_control
        calc reference trajectory in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        """

        # Find the nearest index from where the trajectories are calculated
        _, dist, _, _, ind = nearest_point(np.array([position[0], position[1]]), path[:, (1, 2)])

        reference = self.get_reference_trajectory(np.ones(self.config.TK) * abs(speed), dist, ind, path)

        # TODO: to be improved
        reference[3, :][reference[3, :] - orientation > 5] = \
            np.abs(reference[3, :][reference[3, :] - orientation > 5] - (2 * np.pi))
        reference[3, :][reference[3, :] - orientation < -5] = \
            np.abs(reference[3, :][reference[3, :] - orientation < -5] + (2 * np.pi))

        return reference, 0

    def get_nparray_from_matrix(self, x):
        return np.array(x).flatten()

    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Problem will be solved for every control iteration.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        """
        # Initialize and create vectors for the optimization problem
        self.xk = cvxpy.Variable((self.config.NXK, self.config.TK + 1))  # Vehicle State Vector, 4 x (8 + 1)
        self.uk = cvxpy.Variable((self.config.NU, self.config.TK))  # Control Input vector, 2 x 8
        objective = 0.0  # Objective value of the optimization problem, set to zero
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))  # 4 x 1
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))  # 4 x (8 + 1) ~ Qk + Qfk
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))  # (2 x 2) * 8, diagonal matrix

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))  # (2 x 2) * (8 - 1) ~ 7, difference

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)  # (4 x 4) * 8
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))  # (4 x 4) * (8 + 1)

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # Objective 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        objective += cvxpy.quad_form(cvxpy.vec(self.uk), R_block)
        # cvxpy.vec() - Flattens the matrix X into a vector in column-major order

        # Objective 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep
        #              T weighted by Qf
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)  # Qk + Qfk

        # Objective 3: Difference from one control input to the next control input weighted by Rd
        objective += cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.uk, axis=1)), Rd_block)

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []

        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))  # 4 x (8 + 1)
        input_predict = np.zeros((self.config.NU, self.config.TK + 1))  # 2 x (8 + 1)
        for t in range(self.config.TK):  # 8
            A, B, C = self.model.get_model_matrix(path_predict[:, t], input_predict[:, t])
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))  # A_block changes from list to <class 'scipy.sparse._coo.coo_matrix'>
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)  # 32 x 1

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape  # 32, 32
        self.Annz_k = cvxpy.Parameter(A_block.nnz)  # nnz: number of nonzero elements, nnz = 128
        data = np.ones(self.Annz_k.size)  # 128 x 1, size = 128, all elements are 1
        rows = A_block.row * n + A_block.col  # No. ? element in 32 x 32 matrix
        cols = np.arange(self.Annz_k.size)  # 128 elements that need to be care - diagonal & nonzero, 4 x 4 x 8
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))  # (rows, cols)	data

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data  # real data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")
        # https://www.cvxpy.org/api_reference/cvxpy.atoms.affine.html#cvxpy.reshape

        # B, Same as A
        m, n = B_block.shape  # 32, 16 = 4 x 8, 2 x 8
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)  # nnz = 64
        data = np.ones(self.Bnnz_k.size)  # 64 = (4 x 2) x 8
        rows = B_block.row * n + B_block.col  # No. ? element in 32 x 16 matrix
        cols = np.arange(self.Bnnz_k.size)  # 0, 1, ... 63
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))  # (rows, cols)	data

        # sparse version instead of the old B_block
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")

        # real data
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # Add dynamics constraints to the optimization problem
        constraints += [cvxpy.vec(self.xk[:, 1:])
                        == self.Ak_ @ cvxpy.vec(self.xk[:, :-1]) + self.Bk_ @ cvxpy.vec(self.uk) + (self.Ck_)]
        # cvxpy.vec() - Flattens the matrix X into a vector in column-major order

        # Constraint 2: initial state - set x[k=0] as x0
        constraints += [self.xk[:, 0] == self.x0k]

        # Create the constraints (upper and lower bounds of states and inputs) for the optimization problem
        state_constraints, input_constraints, input_diff_constraints = self.model.get_model_constraints()

        for i in range(self.config.NXK):  # Constraint 3: state constraints
            constraints += [state_constraints[0, i] <= self.xk[i, :], self.xk[i, :] <= state_constraints[1, i]]

        for i in range(self.config.NU):  # Constraint 4: input constraints
            constraints += [input_constraints[0, i] <= self.uk[i, :], self.uk[i, :] <= input_constraints[1, i]]
            constraints += [input_diff_constraints[0, i] <= cvxpy.diff(self.uk[i, :]),
                            cvxpy.diff(self.uk[i, :]) <= input_diff_constraints[1, i]]

        # Create the optimization problem in CVXPY and setup the workspace
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)  # minimize the objective function

    def mpc_prob_solve(self, ref_traj, path_predict, x0, input_predict):
        self.x0k.value = x0

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.model.get_model_matrix(path_predict[:, t], input_predict[:, t])
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.MPC_prob.solve(solver=cvxpy.OSQP, polish=True, adaptive_rho=True, rho=0.01, eps_abs=0.0005,
                            eps_rel=0.0005, verbose=False, warm_start=True)
        # verbose shows the log, other params limit the accuracy and iterations
        # we don't need the extreme precision to 10e-6 with the tolerance of 10000 iterations

        if self.MPC_prob.status == cvxpy.OPTIMAL or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE:
            o_states = self.xk.value
            ou = self.uk.value

        else:
            print("Error: Cannot solve KS mpc... Status : ", self.MPC_prob.status)
            ou, o_states = np.zeros(self.config.NU) * np.NAN, np.zeros(self.config.NXK) * np.NAN

        return ou, o_states

    def linear_mpc_control(self, ref_path, x0, ref_control_input):
        """
        MPC control with updating operational point iteratively
        """

        if np.isnan(ref_control_input).any():
            ref_control_input = np.zeros((2, self.config.TK))

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        state_prediction, input_prediction = self.model.predict_motion(x0, ref_control_input)

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_input_output, mpc_states_output = self.mpc_prob_solve(ref_path, state_prediction, x0, input_prediction)

        return mpc_input_output, mpc_states_output, state_prediction

    def MPC_Control(self, x0, path):  # input current vehicle state and waypoints (== path)
        # Calculate the next reference trajectory for the next T steps
        speed, orientation, position = self.model.get_general_states(x0)
        ref_path, self.target_ind = self.calc_ref_trajectory(position, orientation, speed, path)

        # Solve the Linear MPC Control problem
        self.input_o, states_output, state_predict = self.linear_mpc_control(ref_path, x0, self.input_o)

        # Steering Output: First entry of the MPC steering angle output vector in degree
        u = self.input_o[:, 0]
        steering = u[1]
        speed = u[0] * self.config.DTK + x0[2]  # speed must add the base speed v = v0 + a * dt

        ox = states_output[0]
        oy = states_output[1]

        return steering, speed, ref_path[0], ref_path[1], state_predict[0], state_predict[1], ox, oy
