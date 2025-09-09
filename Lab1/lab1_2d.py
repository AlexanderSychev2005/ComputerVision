import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

matplotlib.use("TkAgg")


def create_pentagon(center=(0, 0), radius=1):
    """
    Create the vertices of a regular pentagon. The pentagon is build around the center points with a given radius.
    """
    angles = np.linspace(0, 2 * np.pi, 6)  # angles for five vertices from 0 to 2pi, like circle
    x = center[0] + radius * np.cos(angles)  # x = r*cos(a) horizontal
    y = center[1] + radius * np.sin(angles)  # y = r*sin(a) vertical
    ones = np.ones_like(x)  # homogeneous coordinates, add a row of ones, to make it possible to do matrix multiplication
    return np.vstack([x, y, ones])  # 3 x 5 matrix, (x, y, 1)


def scale_matrix(sx, sy):
    """
    Create a scaling matrix.

    Parameters:
    sx: scaling factor in x direction
    sy: scaling factor in y direction

    Returns:
    3x3 scaling matrix for scaling
    """
    return np.array([[sx, 0, 0],
                     [0, sy, 0],
                     [0, 0, 1]])


def translation_matrix(tx, ty):
    """
    Create a translation matrix.

    Parameters:
    tx: translation in x direction
    ty: translation in y direction

    Returns:
    3x3 translation matrix for translating
    """
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]])


def rotation_matrix(angle):
    """
    Create a rotation matrix.

    Parameters:
    angle: rotation angle in radians

    Returns:
    3x3 rotation matrix for rotating
    """
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0],
                     [sin, cos, 0],
                     [0, 0, 1]])


def apply_transformation(polygon, transformation_matrix):
    """
    Apply only one transformation to the polygon.
    """
    return transformation_matrix @ polygon  # matrix multiplication


def all_transforming(polygon, S, R, T):
    """
    Apply all transformations: scaling, rotation and translation to the polygon in this order.
    """
    return T @ R @ S @ polygon  # apply scaling, rotation and after that translation, otherwise it doesn't work and the result will be different


polygon = create_pentagon()
print("The polygon matrix:\n", np.round(polygon, 2))

S1 = scale_matrix(1.25, 1.25)  # scale up by 1.25 in x and y
T1 = translation_matrix(1.5, 1.5)  # translate by (1.5, 1.5)
R1 = rotation_matrix(np.radians(15))  # rotate by 15 degrees

fig, ax = plt.subplots(figsize=(6, 6))

ax.set_aspect('equal')

current = polygon.copy()

# colors = plt.cm.viridis(np.linspace(0, 1, 5))
colors = colormaps['viridis'](np.linspace(0, 1, 5))  # Pick 5 colors from the viridis colormap
trajectory = []

for colour in colors:
    ax.set_xlim(-3, 15)
    ax.set_ylim(-3, 15)
    ax.set_aspect('equal')

    center = (np.mean(current[0]), np.mean(current[1]))
    trajectory.append(center)

    ax.plot(np.append(current[0], current[0][0]),
            np.append(current[1], current[1][0]), color=colour)
    plt.pause(0.5)
    current = all_transforming(current, R1, S1, T1)  # Apply all transformations
    print("Polygon after all transformations:\n", np.round(current, 2))

trajectory = np.array(trajectory)
print("Trajectory of centers:\n", np.round(trajectory, 2))
ax.plot(trajectory[:, 0], trajectory[:, 1], 'k--', label='Trajectory')
ax.legend()
plt.show()
