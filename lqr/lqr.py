"""
    MEAM 517 Final Project - LQR Steering Control - LQR class
    Author: Derek Zhou & Tancy Zhao
"""


class Waypoints:
    pass


class CarState:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


class ModelState:

    def __init__(self, e_l, e_l_dot, θ, θ_dot):
        self.e_l = e_l
        self.e_l_dot = e_l_dot
        self.θ = θ
        self.θ_dot = θ_dot

