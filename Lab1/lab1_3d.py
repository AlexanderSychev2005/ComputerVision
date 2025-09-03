import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

matplotlib.use("TkAgg")

colors = colormaps['viridis'](np.linspace(0, 1, 3))
angles_axonometric = [10, 30, 45]
angles_rotation = np.arange(0, 90)
print(angles_rotation)


def create_parallelepiped(center=(0, 0, 0), size=(1, 1, 1)):
    cx, cy, cz = center
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
    # 8 vertices of a parallelepiped
    vertices = np.array([
        [cx - sx, cx + sx, cx + sx, cx - sx, cx - sx, cx + sx, cx + sx, cx - sx],
        [cy - sy, cy - sy, cy + sy, cy + sy, cy - sy, cy - sy, cy + sy, cy + sy],
        [cz - sz, cz - sz, cz - sz, cz - sz, cz + sz, cz + sz, cz + sz, cz + sz],
        [1, 1, 1, 1, 1, 1, 1, 1]  # homogeneous coordinates, (x, y, z, 1)
    ])
    return vertices  # 4 x 8 matrix


def rotation_matrix_z(angle):
    """Create a rotation matrix around the Z-axis."""
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0, 0],
                     [sin, cos, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def axonometric_projection(vertices, angle):
    angle = np.radians(angle)
    x = vertices[0] - vertices[2] * np.cos(angle)
    y = vertices[1] - vertices[2] * np.sin(angle)
    return x, y


parallelepiped = create_parallelepiped()
print(parallelepiped)

fig, axes = plt.subplots(1, 3, figsize=(12, 6))

#
# for ax, angle, color in zip(axes, angles, colors):
#     ax.set_xlim(-2, 2)
#     ax.set_ylim(-2, 2)
#     edges = [
#         (0, 1), (1, 2), (2, 3), (3, 0),  # lower edge
#         (4, 5), (5, 6), (6, 7), (7, 4),  # higher edge
#         (0, 4), (1, 5), (2, 6), (3, 7)  # vertical edges
#     ]
#     x, y = axonometric_projection(parallelepiped, angle)
#     ax.set_title(f'Axonometric Projection at {angle}°')
#     for i, j in edges:
#         ax.plot([x[i], x[j]], [y[i], y[j]], color=color)
#
# plt.pause(2)
# for ax in axes:
#     ax.clear()  # clear axes for next drawing
# # Rotation
# R = rotation_matrix_z(np.radians(30))  # rotate by 30 degrees
# print(np.radians(30))
# print(R)
#
# parallelepiped = R @ parallelepiped  # apply rotation
# print(parallelepiped)
#
# # fig, axes = plt.subplots(1, 3, figsize=(12, 6))
# fig.suptitle('After Rotation by 30° around Z-axis', fontsize=16)
# for ax, angle, color in zip(axes, angles, colors):
#     ax.set_xlim(-2, 2)
#     ax.set_ylim(-2, 2)
#     edges = [
#         (0, 1), (1, 2), (2, 3), (3, 0),  # lower edge
#         (4, 5), (5, 6), (6, 7), (7, 4),  # higher edge
#         (0, 4), (1, 5), (2, 6), (3, 7)  # vertical edges
#     ]
#     x, y = axonometric_projection(parallelepiped, angle)
#     ax.set_title(f'Axonometric Projection at {angle}°')
#
#     for i, j in edges:
#         ax.plot([x[i], x[j]], [y[i], y[j]], color=color)

while True:
    for angle in angles_rotation:
        plt.pause(0.5)
        for ax in axes:
            ax.clear()
        R = rotation_matrix_z(np.radians(angle))  # rotate by angle degrees
        parallelepiped = R @ parallelepiped
        x, y = axonometric_projection(parallelepiped, angle)
        plt.suptitle(f'After Rotation by {angle}° around Z-axis', fontsize=16)

        for ax, ax_angle, color in zip(axes, angles_axonometric, colors):
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # lower edge
                (4, 5), (5, 6), (6, 7), (7, 4),  # higher edge
                (0, 4), (1, 5), (2, 6), (3, 7)  # vertical edges
            ]
            x, y = axonometric_projection(parallelepiped, ax_angle)
            ax.set_title(f'Axonometric Projection at {ax_angle}°')

            for i, j in edges:
                ax.plot([x[i], x[j]], [y[i], y[j]], color=color)

    plt.show()
