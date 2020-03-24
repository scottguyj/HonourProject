import os
from math import sin, radians, degrees, copysign

import numpy as np
import pygame
from pygame.math import Vector2
import math
import pickle
import time

from Car import Car


class Game:

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Honours Project")
        width = 1500
        height = 1000
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False

    def cal_angle(self, x1, y1, x2, y2):
        delta_x = x2 - x1
        delta_y = y2 - y1

        degrees_temp = np.math.atan2(delta_x, delta_y) / math.pi * 180

        return degrees_temp

    def show_text(self, text, x, y):
        font = pygame.font.Font(None, 36)
        text = font.render(text, 1, (255, 255, 255))
        self.screen.blit(text, (x, y))

    def cal_distance(self, x1, y1, x2, y2):
        distance = round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 5)
        return distance

    def render_information(self, distance, crash_counter, current_state, episode, velocity, velocity1):
        distance_text = "Distance = " + str(distance)
        self.show_text(distance_text, 0, 0)

        crash_counter_text = "Fails = " + str(crash_counter)
        self.show_text(crash_counter_text, 0, 20)

        current_state_text = "State: " + current_state
        self.show_text(current_state_text, 0, 40)

        current_episode_text = "Episode: " + str(episode + 1)
        self.show_text(current_episode_text, 0, 60)

        current_velocity_text = "Velocity: " + str(velocity)
        self.show_text(current_velocity_text, 0, 80)

        # ---------------------------------------------------------------

        current_velocity_text = "Velocity: " + str(velocity1)
        self.show_text(current_velocity_text, 400, 80)

        pygame.display.flip()

        self.clock.tick(self.ticks)

    def training_screen(self, episode, distance):
        self.screen.fill((0, 0, 0))
        episode_text = "Training episodes: " + str(episode + 1) + " - " + str((episode + distance) - 1)
        self.show_text(episode_text, 0, 0)

        pygame.display.flip()
        self.clock.tick(self.ticks)

    def run(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "car.png")
        car_image = pygame.image.load(image_path)
        ppu = 32

        crash_counter = 0

        # AI state
        learning_state = True

        # file name goes here for existing q table
        speed_q_table = "qtable-1583233039.pickle"

        steering_q_table = "qtableTstSteering-1583833326.pickle"

        if speed_q_table is None:
            print("Train the speed first")
        else:
            with open(speed_q_table, "rb") as f:
                q_table_speed = pickle.load(f)

        if steering_q_table is None:
            print("Train the steering first")
        else:
            with open(steering_q_table, "rb") as f:
                q_table_steering = pickle.load(f)

        while not self.exit:

            if learning_state:
                state = "Exploring"

                for episode in range(10):

                    lead_car = Car(10, 12)
                    follow_car = Car(5, 12)
                    show = True

                    for i in range(15000):
                        dt = self.clock.get_time() / 1000

                        # Event queue
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                self.exit = True

                        # speed_obs = int((self.cal_distance(lead_car.position_middle.x, lead_car.position_middle.y,
                        #                                   follow_car.position_fmiddle.x,
                        #                                   follow_car.position_fmiddle.y) * 1000))

                        # speed_action = np.argmax(q_table_speed[speed_obs - 2000])
                        pressed = pygame.key.get_pressed()

                        if pressed[pygame.K_UP]:
                            lead_car.action(0, dt)
                            follow_car.action(0, dt)
                        elif pressed[pygame.K_DOWN]:
                            lead_car.action(1, dt)
                            follow_car.action(1, dt)
                        elif pressed[pygame.K_SPACE]:
                            lead_car.action(4, dt)
                        else:
                            lead_car.action(2, dt)
                            follow_car.action(2, dt)

                        lead_car.acceleration = max(-lead_car.max_acceleration,
                                                    min(lead_car.acceleration, lead_car.max_acceleration))
                        follow_car.acceleration = max(-lead_car.max_acceleration,
                                                      min(lead_car.acceleration, lead_car.max_acceleration))
                        if pressed[pygame.K_RIGHT]:
                            lead_car.action(4, dt)
                        elif pressed[pygame.K_LEFT]:
                            lead_car.action(3, dt)
                        else:
                            lead_car.action(5, dt)
                        lead_car.steering = max(-lead_car.max_steering, min(lead_car.steering, lead_car.max_steering))

                        lead_car.update(dt)
                        follow_car.update(dt)

                        # -------------------Steering-------------------
                        if follow_car.velocity.x != 0:

                            angle = self.cal_angle(follow_car.position_fmiddle.x, follow_car.position_fmiddle.y,
                                                   lead_car.position.x, lead_car.position.y)

                            pygame.draw.rect(self.screen, (255, 0, 0),
                                               (lead_car.position_fmiddle.x * ppu, lead_car.position_fmiddle.y * ppu, 5, 5))

                            # print(angle)
                            # print((follow_car.angle + 90))
                            steering_obs = int(round(angle - (follow_car.angle + 90), 2) * 100)

                            if - 18000 > steering_obs > - 37000:
                                steering_obs += 36000
                            print(steering_obs)

                            # print("leading car")
                            # print(lead_car.angle)

                            action = np.argmax(q_table_steering[steering_obs + 9000])

                            # If vehicle is traveling backwards then reverse the steering
                            if follow_car.velocity.x < 0:
                                if action == 0:
                                    action = 1
                                elif action == 1:
                                    action = 0
                                else:
                                    action = 2

                            follow_car.action(action + 3, dt)
                            # print(steering_obs)

                            if action == 0:
                                print("left")
                            elif action == 1:
                                print("right")
                            else:
                                print("Straight")

                            lead_car.update(dt)
                            follow_car.update(dt)

                            angle = self.cal_angle(follow_car.position_fmiddle.x, follow_car.position_fmiddle.y,
                                                   lead_car.position.x, lead_car.position.y)

                            steering_new_obs = int(round(angle - (follow_car.angle + 90), 2) * 100)

                        # --------------------------------------User input--------------------------------------

                        # Controls the steering of the vehicle

                        if show:
                            self.screen.fill((0, 0, 0))
                            rotated_lead = pygame.transform.rotate(car_image, lead_car.angle)
                            rotated_following = pygame.transform.rotate(car_image, follow_car.angle)

                            rect_lead = rotated_lead.get_rect()
                            rect_follow = rotated_following.get_rect()

                            self.screen.blit(rotated_lead,
                                             lead_car.position * ppu - (rect_lead.width / 2, rect_lead.height / 2))
                            self.screen.blit(rotated_following,
                                             follow_car.position * ppu - (rect_follow.width / 2, rect_follow.height /
                                                                          2))

                            self.render_information(0, crash_counter, state, 0, lead_car.velocity.x,
                                                    follow_car.velocity.x)

                            pygame.display.update()

                            self.clock.tick(self.ticks)
                    print(follow_car.angle)

                with open(f"qtableSteering-{int(time.time())}.pickle", "wb") as f:
                    pickle.dump(q_table_steering, f)
                    learning_state = False
                self.exit = True

        pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.run()
