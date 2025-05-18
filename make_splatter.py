import numpy as np
import os
# import matplotlib.pyplot as plt
# Creates a numpy array (x, y, time) of random walks smushed together. Saves to /data/splatter.npy

os.makedirs("data", exist_ok=True)

def make_random_walk(): 
    """returns numpy array of random length which is a random walk of x, y, and time in seconds"""
    num_steps = np.random.randint(100, 1001)

    start_pos = np.random.randint(-10, 10, size=2)
    steps = np.random.choice([-1, 0, 1], size=(num_steps, 2))
    positions = np.cumsum(steps, axis=0) + start_pos
    time = np.arange(num_steps)
    
    # Combine time, x, y into single array
    result = np.column_stack((positions, time))
    return result

N = 10
walks = []
for i in range(N): 
    walk = make_random_walk()
    np.save(f"data/walk_{i}.npy", walk)
    walks.append(walk)


walks = np.concat(walks)
splatter = walks[walks[:, 2].argsort()] # sort by time
np.save("data/splatter.npy", splatter)