import numpy as np
from f1tenth_gym.MPC.helpers.closest_point import get_closest_point_vectorized, nearest_point_on_trajectory
from pyglet.gl import GL_POINTS
import cvxpy


def draw_point(e, point, colour):
    scaled_point = 50. * point
    obj = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_point[0], scaled_point[1], 0]), ('c3B/stream', colour))
    return obj


class MPC:
    def __init__(self, model):
        self.time_step = 0.1
        self.preiction_horizon = 10
        self.closest_traj_point = np.array([0, 0])
        self.reference_traj_show = np.array([[0, 0]])
        self.predicted_traj_show = np.array([[0, 0]])
        self.dyn_obj_drawn = []
        self.point_dist_waypoints = 0
        self.model = model

        # optimization parameters
        self.Q = np.diag([32.0, 32.0, 5.0, 0.0])
        self.QN = np.diag([32.0, 32.0, 5.0, 0.0])
        self.R = np.diag([0.01, 0.5])
        self.RD = np.diag([0.01, 0.3])

    def load_waypoints_tum(self, conf):
        """
        loads waypoints
        Load waypoints that are in the same format as output from TUM optimizer
        """
        # waypoints =  [s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2]
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
        self.waypoints_distances = np.linalg.norm(self.waypoints[1:, (1, 2)] - self.waypoints[:-1, (1, 2)], axis=1)
        self.point_dist_waypoints = np.linalg.norm(self.waypoints[1, (1, 2)] - self.waypoints[0, (1, 2)])

        self.waypoints[:, 3] = (self.waypoints[:, 3] + np.pi) % (2 * np.pi) - np.pi

    def find_closest_point(self, position):
        # later can be changed to dynamic search based on last known position
        # idx = get_closest_point_vectorized(position, self.waypoints[:, (1, 2)])
        closest_traj_point, _, t, dist_from_segment_start, idx = nearest_point_on_trajectory(position,
                                                                                             self.waypoints[:, (1, 2)])
        return closest_traj_point, dist_from_segment_start, idx, t

    def find_closest_point_v2(self, position):
        # later can be changed to dynamic search based on last known position
        idx = get_closest_point_vectorized(position, self.waypoints[:, (1, 2)])
        # not finished

    def draw_debug(self, e):
        # delete dynamic objects
        while len(self.dyn_obj_drawn) > 0:
            self.dyn_obj_drawn[0].delete()
            self.dyn_obj_drawn.pop(0)

        # spawn new objects

        for p in self.reference_traj_show:
            self.dyn_obj_drawn.append(draw_point(e, p, [255, 0, 0]))

        for p in self.predicted_traj_show:
            self.dyn_obj_drawn.append(draw_point(e, p, [0, 255, 0]))

    def predict_speed(self, current_speed):
        speeds = np.ones((self.preiction_horizon,)) * current_speed
        return speeds

    def get_reference_trajectory(self, predicted_speeds, dist_from_segment_start, idx):
        s_relative = np.zeros((self.preiction_horizon + 1,))
        s_relative[0] = dist_from_segment_start
        s_relative[1:] = predicted_speeds * self.time_step
        s_relative = np.cumsum(s_relative)

        waypoints_distances_relative = np.cumsum(np.roll(self.waypoints_distances, -idx))

        index_relative = np.int_(np.ones((self.preiction_horizon + 1,)))
        for i in range(self.preiction_horizon + 1):
            index_relative[i] = (waypoints_distances_relative <= s_relative[i]).sum()
        index_absolute = np.mod(idx + index_relative, self.waypoints.shape[0] - 1)

        segment_part = s_relative - (
                waypoints_distances_relative[index_relative] - self.waypoints_distances[index_absolute])

        t = (segment_part / self.waypoints_distances[index_absolute])
        # print(np.all(np.logical_and((t < 1.0), (t > 0.0))))

        position_diffs = (self.waypoints[np.mod(index_absolute + 1, self.waypoints.shape[0] - 1)][:, (1, 2)] -
                          self.waypoints[index_absolute][:, (1, 2)])

        orientation_diffs = (self.waypoints[np.mod(index_absolute + 1, self.waypoints.shape[0] - 1)][:, 3] -
                             self.waypoints[index_absolute][:, 3])

        speed_diffs = (self.waypoints[np.mod(index_absolute + 1, self.waypoints.shape[0] - 1)][:, 5] -
                       self.waypoints[index_absolute][:, 5])

        interpolated_positions = self.waypoints[index_absolute][:, (1, 2)] + (t * position_diffs.T).T

        interpolated_orientations = self.waypoints[index_absolute][:, 3] + (t * orientation_diffs)
        interpolated_orientations = (interpolated_orientations + np.pi) % (2 * np.pi) - np.pi

        interpolated_speeds = self.waypoints[index_absolute][:, 5] + (t * speed_diffs)

        reference = np.array([
            interpolated_positions[:, 0],
            interpolated_positions[:, 1],
            interpolated_orientations,
            interpolated_speeds
        ])

        return reference

    def create_optimization_problem(self, x_init, xr, x_predicted):

        # Init
        x = cvxpy.Variable((self.model.number_of_states, self.preiction_horizon + 1))
        u = cvxpy.Variable((2, self.preiction_horizon))
        constraints = [x[:, 0] == x_init]
        objective = 0.0

        for k in range(self.preiction_horizon):
            objective += cvxpy.quad_form(x[:, k] - xr[:, k], self.Q)
            objective += cvxpy.quad_form(u[:, k], self.R)
            A, B, C = self.model.get_linearized_f(x_predicted[:, k], np.array([0, 0]))

            constraints += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k] + C]

            constraints += [0.0 <= x[3, k], x[3, k] <= 45.0]
            constraints += [[-40, -0.910] <= u[:, k], u[:, k] <= [40, 0.910]]

        objective += cvxpy.quad_form(x[:, self.preiction_horizon] - xr[:, self.preiction_horizon], self.QN)

        prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
        return prob

    def get_drive_command(self, state, control_input_temp):
        self.model.update_state(state)
        speed, orientation, position = self.model.get_general_states()

        self.closest_traj_point, dist_from_segment_start, idx, _ = self.find_closest_point(position)
        predicted_speeds = self.predict_speed(speed)

        reference = self.get_reference_trajectory(predicted_speeds, dist_from_segment_start, idx)

        self.reference_traj_show = reference[(0, 1), :].T

        # print(str(self.waypoints[idx, 3]) + "   " + str(state[2]) + "   " + str(interpolated_orientations[0]))

        predicted_trajectory = self.model.predict_trajectory(np.ones((self.preiction_horizon, 2)) * control_input_temp,
                                                             self.time_step)
        self.predicted_traj_show = predicted_trajectory

        prob = self.create_optimization_problem(self.model.state, reference, predicted_trajectory.T)
        prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)
