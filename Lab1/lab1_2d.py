import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
matplotlib.use("TkAgg")


def create_pentagon(center=(0, 0), radius=1):
    """Create the vertices of a pentagon."""
    angles = np.linspace(0, 2 * np.pi, 6)[:-1]  # angles for five vertices from 0 to 2pi, circle
    x = center[0] + radius * np.cos(angles)  # x = r*cos(a)
    y = center[1] + radius * np.sin(angles)  # y = r*sin(a)
    ones = np.ones_like(x)  # homogeneous coordinates
    return np.vstack([x, y, ones])  # 3 x 5 matrix, (x, y, 1)


def scale_matrix(sx, sy):
    """Create a scaling matrix."""
    return np.array([[sx, 0, 0],
                     [0, sy, 0],
                     [0, 0, 1]])


def translation_matrix(tx, ty):
    """Create a translation matrix."""
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]])


def all_transforming(polygon, S, T):
    return T @ S @ polygon  # apply scaling then translation


polygon = create_pentagon()

S1 = scale_matrix(1.25, 1.25)  # scale up by 1.5
T1 = translation_matrix(1.5, 1.1)  # translate by (2,

fig, ax = plt.subplots(figsize=(6, 6))

ax.set_aspect('equal')
print(polygon)
current = polygon.copy()

colors = plt.cm.viridis(np.linspace(0, 1, 5))
colors = colormaps['viridis'](np.linspace(0, 1, 5))
trajectory = []

for colour in colors:
    ax.set_xlim(-3, 12)
    ax.set_ylim(-3, 12)
    ax.set_aspect('equal')

    center = (np.mean(current[0]), np.mean(current[1]))
    trajectory.append(center)

    ax.plot(np.append(current[0], current[0][0]),
            np.append(current[1], current[1][0]), color=colour)
    plt.pause(1)
    current = all_transforming(current, S1, T1)

trajectory = np.array(trajectory)
ax.plot(trajectory[:, 0], trajectory[:, 1], 'k--', label='Trajectory')
plt.show()
