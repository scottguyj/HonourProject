import pickle
import time
import numpy as np
steering_q_table_1 = "qtableSteering-1584383836.pickle"
steering_q_table_2 = "qtableTstSteering-1583833326.pickle"

counter = 0

if steering_q_table_1 is None:
    q_table_steering = np.zeros((30000, 3))
else:
    with open(steering_q_table_1, "rb") as f:
        q_table_steering1 = pickle.load(f)

if steering_q_table_2 is None:
    q_table_steering = np.zeros((30000, 3))
else:
    with open(steering_q_table_2, "rb") as f:
        q_table_steering2 = pickle.load(f)

for i in range(1):
    action1 = np.argmax(q_table_steering1[1738])
    action2 = np.argmax(q_table_steering2[1738])
    print(q_table_steering1[1738])
    print(action1)
    print(action2)
