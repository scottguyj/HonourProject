import pygame
import numpy as np
# matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import math
from Car import Car
from Enviroment import Game
import keyboard

env = Game()
show_loading = False

crash_counter = 0

DISTANCE_MIN = 0
DISTANCE_MAX = 60
DISTANCE_IDEAL = 40

# rewards
HM_EPISODES = 500
CRASH_PENALTY = 300
DISTANCE_REWARD = 5000

# q learning Variables
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 100
LEARNING_RATE = 0.1
DISCOUNT = 0.95

# AI state
learning_state = True
state = ""
end_program = False

# file name goes here for existing q table
start_q_table = None #"qtable-1580999182.pickle"

if start_q_table is None:
    q_table = np.zeros((100, 3))
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)
        print("Q table Found")

        # This means that the AI has already been trained and can move onto the demo
        learning_state = False


episode_rewards = []
while not end_program:
    if learning_state:
        state = "Exploring"

        for episode in range(HM_EPISODES):
            lead_car = Car(150, 100)
            following_car = Car(150, 150)

            if episode % SHOW_EVERY == 0:
                print(f"on # {episode}, epsilon: {epsilon}")
                show = True
                show_loading = True
            else:
                show = False

            episode_rewards = 0

            for i in range(3000):

                while not end_program:
                    obs = int(math.sqrt((following_car.position_x - lead_car.position_x) ** 2 +
                                        (following_car.position_y - lead_car.position_y) ** 2)) - 40
                    if np.random.random() > epsilon:
                        action = np.argmax(q_table[obs])
                    else:
                        action = np.random.randint(0, 3)

                    following_car.action(action)

                    new_obs = int(math.sqrt((following_car.position_x - lead_car.position_x) ** 2 +
                                            (following_car.position_y - lead_car.position_y) ** 2)) - 40

                    if new_obs < DISTANCE_MIN or new_obs > DISTANCE_MAX:
                        reward = -CRASH_PENALTY
                        crash_counter += 1
                    elif new_obs == DISTANCE_IDEAL:
                        reward = DISTANCE_REWARD
                    else:
                        reward = -1

                    # Logic Leading Car
                    if keyboard.is_pressed('w'): # press 'w' to move teh leading vehicle forward
                        lead_car.action(1)
                    elif keyboard.is_pressed('s'): # press 's' to move the leading vehicle backward
                        lead_car.action(0)
                    elif keyboard.is_pressed('q'): # Press 'q' to quit the program
                        end_program = True
                    else:
                        lead_car.action(2)

                    max_future_q = np.max(q_table[new_obs])
                    current_q = q_table[obs][action]

                    if reward == DISTANCE_REWARD:
                        new_q = DISTANCE_REWARD
                    elif reward == -CRASH_PENALTY:
                        new_q = -CRASH_PENALTY
                    else:
                        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                    q_table[obs][action] = new_q

                    if show:
                        env.render_cars(lead_car.position_x, lead_car.position_y, following_car.position_x,
                                        following_car.position_y,
                                        new_obs, crash_counter, state, episode)
                    elif show_loading:
                        env.training_screen(episode, SHOW_EVERY)
                        show_loading = False

                    episode_rewards += reward
                    if reward == -CRASH_PENALTY:
                        break

        with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
            pickle.dump(q_table, f)
            learning_state = False

    if not learning_state:
        print("Started exploiting")
        lead_car = Car(150, 100)
        following_car = Car(150, 150)

        # This means that there is no random actions being made
        epsilon = 0
        state = "Exploiting"

    while not end_program:
        obs = int(math.sqrt((following_car.position_x - lead_car.position_x) ** 2 +
                            (following_car.position_y - lead_car.position_y) ** 2)) - 40
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 3)

        following_car.action(action)

        new_obs = int(math.sqrt((following_car.position_x - lead_car.position_x) ** 2 +
                                    (following_car.position_y - lead_car.position_y) ** 2)) - 40
        # Logic Leading Car
        if keyboard.is_pressed('w'):
            lead_car.action(1)
        elif keyboard.is_pressed('s'):
            lead_car.action(0)
        elif keyboard.is_pressed('q'):
            end_program = True
        else:
            lead_car.action(2)

        env.render_cars(lead_car.position_x, lead_car.position_y, following_car.position_x,
                        following_car.position_y,
                        new_obs, crash_counter, state, -2)
env.close()
