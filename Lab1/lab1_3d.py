import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

matplotlib.use("TkAgg")

colors = colormaps['viridis'](np.linspace(0, 1, 3))
angles_axonometric = [(35.264, 45), (30, 45), (15, 30)]  # (angle_x, angle_y)
angles_rotation = np.arange(0, 90)


def create_parallelepiped(center=(0, 0, 0), size=(1, 1, 1)):
    """
    Create the vertices of a parallelepiped. The parallelepiped is built around the center point with a given size.
    """
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


def rotation_matrix_x(angle):
    """
    Create a rotation matrix around the X-axis.
    """
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0, 0],
        [0, cos, -sin, 0],
        [0, sin, cos, 0],
        [0, 0, 0, 1]
    ])


def rotation_matrix_y(angle):
    """
    Create a rotation matrix around the Y-axis.
    """
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([
        [cos, 0, sin, 0],
        [0, 1, 0, 0],
        [-sin, 0, cos, 0],
        [0, 0, 0, 1]
    ])


def rotation_matrix_z(angle):
    """Create a rotation matrix around the Z-axis."""
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0, 0],
                     [sin, cos, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def projection_matrix_xy():
    """Create an orthographic projection matrix onto the XY plane."""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 1]])


def axonometric_matrix(a, b):
    """Create an axonometric projection matrix given angles a and b in degrees."""
    a = np.radians(a)
    b = np.radians(b)

    Rx = rotation_matrix_x(a)
    Ry = rotation_matrix_y(b)
    P = projection_matrix_xy()
    return P @ Ry @ Rx  # First - rotate around x, then around y, then project


parallelepiped = create_parallelepiped()
print("Original vertices:\n", parallelepiped)

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

fig, axes = plt.subplots(1, 3, figsize=(12, 6))

for angle in angles_rotation:
    plt.pause(0.3)
    for ax in axes:
        ax.clear()

    plt.suptitle(f'After Rotation by {angle}° around Z-axis', fontsize=16)

    Rz = rotation_matrix_z(np.radians(angle))  # rotate by angle degrees
    print("-" * 40, "\n")
    print("Angle (degrees):", angle, "Angle (radians):", np.radians(angle))
    print("Rotation Matrix:\n", Rz)
    for ax, ax_angle, color in zip(axes, angles_axonometric, colors):

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # lower edge
            (4, 5), (5, 6), (6, 7), (7, 4),  # higher edge
            (0, 4), (1, 5), (2, 6), (3, 7)  # vertical edges
        ]
        M = axonometric_matrix(ax_angle[0], ax_angle[1])
        print(f"Axonometric angles: {ax_angle}, Axonometric Matrix:\n", M)
        vertices_proj = M @ Rz @ parallelepiped  # apply rotation and then axonometric projection
        print("Projected vertices:\n", np.round(vertices_proj, 2))
        x, y = vertices_proj[0], vertices_proj[1]
        ax.set_title(f'Axonometric Projection at x={ax_angle[0]}°, y={ax_angle[1]}')

        for i, j in edges:
            ax.plot([x[i], x[j]], [y[i], y[j]], color=color)

plt.show()
