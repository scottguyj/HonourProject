import os
from math import sin, radians, degrees, copysign

import pygame
from pygame.math import Vector2


class Car:
    def __init__(self, x, y, angle=0.0, length=4, max_steering=30, max_acceleration=5.0):
        self.position_x = x
        self.position_y = y
        self.velocity = Vector2(0.0, 0.0)

        self.back_current_vel = 0
        self.for_current_vel = 0

        self.angle = angle
        self.length = length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_velocity = 5
        self.brake_deceleration = 10
        self.free_deceleration = 2

        self.acceleration = 0.0
        self.steering = 0.0

    def action(self, act):

        # Forward movement
        if act == 0:
            self.position_y += 1

        # Backward Movement
        elif act == 1:
            self.position_y -= 1

        elif act == 2:
            self.position_y -= 0

    def update(self, dt):
        self.velocity += (self.acceleration * dt, 0)
