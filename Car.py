import os
from math import sin, radians, degrees, copysign
import math

import pygame
from pygame.math import Vector2


class Car:
    def __init__(self, x, y, angle=0.0, length=4, max_steering=30, max_acceleration=5.0):
        self.position = Vector2(x, y)

        # Sensor Co-ordinates
        self.position_middle = Vector2(x, y)
        self.position_left = Vector2(x, y)
        self.position_right = Vector2(x, y)

        self.position_fmiddle = Vector2(x, y)
        self.position_fleft = Vector2(x, y)
        self.position_fright = Vector2(x, y)

        self.velocity = Vector2(0.0, 0.0)
        self.angle = angle
        self.length = length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.max_velocity = 10
        self.brake_deceleration =20
        self.free_deceleration = 5

        self.acceleration = 0.0
        self.steering = 0.0

        self.r_angle_middle = 0.0
        self.r_angle_left = 0.0
        self.r_angle_right = 0.0

        self.r_angle_fmiddle = 0.0
        self.r_angle_fleft = 0.0
        self.r_angle_fright = 0.0

    def update(self, dt):
        self.velocity += (self.acceleration * dt, 0)
        self.velocity.x = max(-self.max_velocity, min(self.velocity.x, self.max_velocity))

        if self.steering:
            turning_radius = self.length / sin(radians(self.steering))
            angular_velocity = self.velocity.x / turning_radius
        else:
            angular_velocity = 0

        self.position += self.velocity.rotate(-self.angle) * dt
        self.position_middle += self.velocity.rotate(-self.angle) * dt
        self.position_left += self.velocity.rotate(-self.angle) * dt
        self.position_right += self.velocity.rotate(-self.angle) * dt

        self.angle += degrees(angular_velocity) * dt

        if self.angle > 360:
            self.angle = -360
        elif self.angle < - 360:
            self.angle = 360

        # Sensor logic when steering
        self.r_angle_middle = (self.angle - 180) * (math.pi/180)
        self.r_angle_left = (self.angle - 205) * (math.pi/180)
        self.r_angle_right = (self.angle - 155) * (math.pi/180)

        self.position_middle.x = self.position.x + (2 * math.cos(-self.r_angle_middle))
        self.position_middle.y = self.position.y + (2 * math.sin(-self.r_angle_middle))

        self.position_left.x = self.position.x + (2 * math.cos(-self.r_angle_left))
        self.position_left.y = self.position.y + (2 * math.sin(-self.r_angle_left))

        self.position_right.x = self.position.x + (2 * math.cos(-self.r_angle_right))
        self.position_right.y = self.position.y + (2 * math.sin(-self.r_angle_right))

        self.r_angle_fmiddle = self.angle * (math.pi / 180)
        self.r_angle_fleft = (self.angle - 25) * (math.pi / 180)
        self.r_angle_fright = (self.angle + 25) * (math.pi / 180)

        self.position_fmiddle.x = self.position.x + (2 * math.cos(-self.r_angle_fmiddle))
        self.position_fmiddle.y = self.position.y + (2 * math.sin(-self.r_angle_fmiddle))

        self.position_fleft.x = self.position.x + (2 * math.cos(-self.r_angle_fleft))
        self.position_fleft.y = self.position.y + (2 * math.sin(-self.r_angle_fleft))

        self.position_fright.x = self.position.x + (2 * math.cos(-self.r_angle_fright))
        self.position_fright.y = self.position.y + (2 * math.sin(-self.r_angle_fright))

    def action(self, act, dt):

        # Forward movement
        if act == 0:
            if self.velocity.x < 0:
                self.acceleration = self.brake_deceleration
            else:
                self.acceleration += 1 * dt

        # Backward Movement
        elif act == 1:
            if self.velocity.x > 0:
                self.acceleration = -self.brake_deceleration
            else:
                self.acceleration -= 1 * dt

        elif act == 2:
            if abs(self.velocity.x) > dt * self.free_deceleration:
                self.acceleration = -copysign(self.free_deceleration, self.velocity.x)
            else:
                if dt != 0:
                    self.acceleration = -self.velocity.x / dt

        elif act == 3:
            self.steering += 30 * dt

        elif act == 4:
            self.steering -= 30 * dt

        elif act == 5:
            if abs(self.velocity.x) > dt * self.brake_deceleration:
                self.acceleration = -copysign(self.brake_deceleration, self.velocity.x)
            else:
                self.acceleration = -self.velocity.x / dt

        elif act == 6:
            self.steering = 0
        self.acceleration = max(-self.max_acceleration, min(self.acceleration, self.max_acceleration))


