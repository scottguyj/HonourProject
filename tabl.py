import pickle
import time
import numpy as np
steering_q_table = None

if steering_q_table is None:
    q_table_steering = np.zeros((30000, 3))
else:
    with open(steering_q_table, "rb") as f:
        q_table_steering = pickle.load(f)


for i in range(8999):
    q_table_steering[i][0] = -1
    q_table_steering[i][2] = -1
    q_table_steering[i][1] = 1

q_table_steering[9000][2] = 5000
q_table_steering[9000][0] = 2000
q_table_steering[9000][1] = 2000

for i in range(8999):
    q_table_steering[i + 9001][0] = 1
    q_table_steering[i + 9001][2] = -1
    q_table_steering[i + 9001][1] = - 1

with open(f"qtableTstSteering-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table_steering, f)
    learning_state = False