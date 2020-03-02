import os
from math import sin, radians, degrees, copysign

import pygame
from pygame.math import Vector2
import math

from Car import Car


class Game:

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Honours Project")
        width = 1000
        height = 1000
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False

        self.crash_counter = 0

        self.DISTANCE_MIN = 2.0
        self.DISTANCE_MAX = 6.0
        self.DISTANCE_IDEAL = 4.0

        # rewards
        self.HM_EPISODES = 500
        self.CRASH_PENALTY = 300
        self.DISTANCE_REWARD = 5000

        # q learning Variables
        self.epsilon = 0.9
        self.EPS_DECAY = 0.9998
        self.SHOW_EVERY = 100
        self.LEARNING_RATE = 0.1
        self.DISCOUNT = 0.95
        self.current_state = "Developing"

        self.end_program = False
        start_q_table = None

        # if start_q_table is None:

    def show_text(self, text, x, y):
        font = pygame.font.Font(None, 36)
        text = font.render(text, 1, (255, 255, 255))
        self.screen.blit(text, (x, y))

    def cal_distance(self, x1, y1, x2, y2):
        distance = round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 3)
        return distance

    def render_information(self, distance, crash_counter, current_state, episode, velocity):
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
        lead_car = Car(10, 5)
        follow_car = Car(5, 5)

        while not self.exit:
            dt = self.clock.get_time() / 1000

            # Event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            # User input
            pressed = pygame.key.get_pressed()

            # Controls the Acceleration, braking and reverse
            if pressed[pygame.K_UP]:
                lead_car.action(0, dt)
            elif pressed[pygame.K_DOWN]:
                lead_car.action(1, dt)
            elif pressed[pygame.K_SPACE]:
                lead_car.action(4, dt)
            else:
                lead_car.action(5, dt)

            lead_car.acceleration = max(-lead_car.max_acceleration, min(lead_car.acceleration, lead_car.max_acceleration))

            # Controls the steering of th vehicle
            if pressed[pygame.K_RIGHT]:
                lead_car.action(2, dt)
            elif pressed[pygame.K_LEFT]:
                lead_car.action(3, dt)
            else:
                lead_car.steering = 0
            lead_car.steering = max(-lead_car.max_steering, min(lead_car.steering, lead_car.max_steering))

            # Logic
            lead_car.update(dt)

            distance_middle = self.cal_distance(lead_car.position_middle.x, lead_car.position_middle.y,
                                                follow_car.position_fmiddle.x, follow_car.position_fmiddle.y)

            if distance_middle < self.DISTANCE_MIN or distance_middle > self.DISTANCE_MAX:
                self.crash_counter += 1

            # Drawing
            self.screen.fill((0, 0, 0))
            rotated_lead = pygame.transform.rotate(car_image, lead_car.angle)
            rotated_following = pygame.transform.rotate(car_image, follow_car.angle)

            rect_lead = rotated_lead.get_rect()
            rect_follow = rotated_following.get_rect()

            self.screen.blit(rotated_lead, lead_car.position * ppu - (rect_lead.width / 2, rect_lead.height / 2))
            self.screen.blit(rotated_following, follow_car.position * ppu - (rect_follow.width / 2, rect_follow.height /
                                                                             2))

            self.render_information(distance_middle,self.crash_counter, self.current_state, 0, lead_car.velocity.x)
            print(lead_car.angle)
            # pygame.draw.rect(self.screen, (255, 0, 0),
            #   (lead_car.position_left.x * ppu, lead_car.position_left.y * ppu, 5, 5))
            # pygame.draw.rect(self.screen, (255, 0, 0),
            #   (lead_car.position_middle.x * ppu, lead_car.position_middle.y * ppu, 5, 5))
            # pygame.draw.rect(self.screen, (255, 0, 0),
            #   (lead_car.position_right.x * ppu, lead_car.position_right.y * ppu, 5, 5))
            # pygame.draw.rect(self.screen, (255, 0, 0),
            #   (lead_car.position_fleft.x * ppu, lead_car.position_fleft.y * ppu, 5, 5))
            # pygame.draw.rect(self.screen, (255, 0, 0),
            #   (lead_car.position_fmiddle.x * ppu, lead_car.position_fmiddle.y * ppu, 5, 5))
            # pygame.draw.rect(self.screen, (255, 0, 0),
            #   (lead_car.position_fright.x * ppu, lead_car.position_fright.y * ppu, 5, 5))

            pygame.display.update()

            self.clock.tick(self.ticks)
        pygame.quit()


if __name__ == '__main__':
    game = Game()
    game.run()
