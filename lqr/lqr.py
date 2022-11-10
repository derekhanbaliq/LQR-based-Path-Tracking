"""
    MEAM 517 Final Project - LQR Steering Control - LQR class
    Author: Derek Zhou & Tancy Zhao
"""


class Waypoint:
    pass


class CarState:

    def __init__(self, x=0.0, y=0.0, θ_h=0.0, v=0.0):
        self.x = x
        self.y = y
        self.θ_h = θ_h
        self.v = v


class LateralKinematicVehicleModelState:

    def __init__(self, e_l, e_l_dot, e_θ, e_θ_dot):
        self.e_l = e_l
        self.e_l_dot = e_l_dot
        self.e_θ = e_θ
        self.e_θ_dot = e_θ_dot

