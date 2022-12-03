"""
    MEAM 517 Final Project - LQR Steering Control - Renderer class
    Author: Derek Zhou & Tancy Zhao
    References: https://github.com/f1tenth/f1tenth_gym/tree/main/examples
"""
import numpy as np
from pyglet.gl import GL_POINTS  # game interface


class Renderer:

    def __init__(self, waypoints):
        self.waypoints = waypoints  # csv data
        self.drawn_waypoints = []
        self.obs = None

        # for MPC debug
        self.reference_traj_show = np.array([[0, 0]])
        self.predicted_traj_show = np.array([[0, 0]])
        self.dyn_obj_drawn = []
        self.f = 0

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        points = np.vstack((self.waypoints[:, 1], self.waypoints[:, 2])).T  # N x 2

        scaled_points = 50. * points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [255, 255, 255]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    def load_obs(self, obs):
        self.obs = obs

    def render_path(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        x = self.obs['poses_x']
        y = self.obs['poses_y']

        point = np.array([x, y])

        scaled_point = 50. * point

        b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_point[0], scaled_point[1], 0.]),
                        ('c3B/stream', [255, 0, 0]))
        self.drawn_waypoints.append(b)

    def draw_debug(self, e):
        # delete dynamic objects
        while len(self.dyn_obj_drawn) > 0:
            if self.dyn_obj_drawn[0] is not None:
                self.dyn_obj_drawn[0].delete()
            self.dyn_obj_drawn.pop(0)

        # spawn new objects
        for p in self.reference_traj_show:
            self.dyn_obj_drawn.append(self.draw_point(e, p, [255, 0, 0]))

        for p in self.predicted_traj_show:
            self.dyn_obj_drawn.append(self.draw_point(e, p, [0, 255, 0]))

    def draw_point(self, e, point, colour):
        scaled_point = 50. * point
        ret = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_point[0], scaled_point[1], 0]),
                          ('c3B/stream', colour))
        return ret
