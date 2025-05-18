import numpy as np
from glob import glob
import matplotlib.pyplot as plt

splatter = np.load("data/splatter.npy")
os.makedirs("plots", exist_ok=True)

# reads in splatter all at once

# point is defined as x, y, time
tracks = [[splatter[0]]]

VELOCITY_THRESHOLD = 2

for i in range(1, len(splatter)):
    next_point = splatter[i]
    x2, y2, time2 = next_point
    last_points_in_tracks = [t[-1] for t in tracks]
    added_to_track = False

    for j, (x1, y1, time1) in enumerate(last_points_in_tracks):
        distance_delta = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        time_delta = time2 - time1

        if time_delta == 0:
            velocity = np.inf
        else:
            velocity = np.abs(distance_delta / time_delta)

        if velocity <= VELOCITY_THRESHOLD and time_delta < 3:
            tracks[j].append(next_point)
            added_to_track = True
            break

    if not added_to_track:
        tracks.append([next_point])


tracks = [np.stack(t) for t in tracks if len(t) > 1]

ground_truths = [np.load(f) for f in glob("data/walk_*.npy")]

print(len(tracks))
print(len(ground_truths))
print(len(tracks) == len(ground_truths))

for gt in ground_truths:
    plt.plot(gt[:, 0], gt[:, 1])
plt.savefig("plots/ground_truths.png")
plt.close()


for pt in tracks:
    plt.plot(pt[:, 0], pt[:, 1])
plt.savefig("plots/predicted_tracks.png")
plt.close()


for i, pt in enumerate(tracks):
    plt.plot(pt[:, 0], pt[:, 1])
    plt.savefig(f"plots/pt_{i}.png")
    plt.close()