# Copyright 2025 Todd Hutchinson, Anonymous
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Proprietary Software - All Rights Reserved
#
# Copyright (C) 2025 Todd Hutchinson
#
# This software is proprietary and confidential. Unauthorized copying,
# distribution, modification, or use is strictly prohibited without
# express written permission from Todd Hutchinson.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# B-spline basis function (helper for green curve)
def bspline_basis(u, i, p, knots):
    if p == 0:
        if i < 0 or i + 1 >= len(knots):
            return 0.0
        return 1.0 if knots[i] <= u <= knots[i + 1] else 0.0
    
    if i < 0 or i >= len(knots) - 1:
        return 0.0
    
    term1 = 0.0
    if i + p < len(knots):
        den1 = knots[i + p] - knots[i]
        if den1 > 0:
            term1 = ((u - knots[i]) / den1) * bspline_basis(u, i, p - 1, knots)
    
    term2 = 0.0
    if i + p + 1 < len(knots):
        den2 = knots[i + p + 1] - knots[i + 1]
        if den2 > 0:
            term2 = ((knots[i + p + 1] - u) / den2) * bspline_basis(u, i + 1, p - 1, knots)
    
    return term1 + term2

# Custom interoperations green curve function (B-spline smoothing, supports open/closed)
def custom_interoperations_green_curve(points, kappas, is_closed=False):
    """Generate smoothed curve using B-spline interpolation.

    Args:
        points: List of [x, y] points to smooth.
        kappas: List of weights for each point.
        is_closed: If True, treat as closed curve (periodic); else open (clamped).

    Returns:
        smooth_x, smooth_y: Arrays of smoothed x and y coordinates.
    """
    points = np.array(points)
    kappas = np.array(kappas)
    degree = 3
    num_output_points = 1000
    
    if is_closed and len(points) > degree:
        n = len(points)
        extended_points = np.concatenate((points[n-degree:], points, points[0:degree]))
        extended_kappas = np.concatenate((kappas[n-degree:], kappas, kappas[0:degree]))
        len_extended = len(extended_points)
        knots = np.linspace(-degree / float(n), 1 + degree / float(n), len_extended + 1)
        
        u_fine = np.linspace(0, 1, num_output_points, endpoint=False)
        
        smooth_x = np.zeros(num_output_points)
        smooth_y = np.zeros(num_output_points)
        
        for j, u in enumerate(u_fine):
            num_x, num_y, den = 0.0, 0.0, 0.0
            for i in range(len_extended):
                b = bspline_basis(u, i, degree, knots)
                w = extended_kappas[i] * b
                num_x += w * extended_points[i, 0]
                num_y += w * extended_points[i, 1]
                den += w
            if den > 0:
                smooth_x[j] = num_x / den
                smooth_y[j] = num_y / den
        
        smooth_x = np.append(smooth_x, smooth_x[0])
        smooth_y = np.append(smooth_y, smooth_y[0])
        
    else:  # Open (clamped) B-spline
        n = len(points)
        knots = np.concatenate(([0] * (degree + 1), np.linspace(0, 1, n - degree + 1)[1:-1], [1] * (degree + 1)))
        
        u_fine = np.linspace(0, 1, num_output_points)
        
        smooth_x = np.zeros(num_output_points)
        smooth_y = np.zeros(num_output_points)
        
        for j, u in enumerate(u_fine):
            num_x, num_y, den = 0.0, 0.0, 0.0
            for i in range(n):
                b = bspline_basis(u, i, degree, knots)
                w = kappas[i] * b
                num_x += w * points[i, 0]
                num_y += w * points[i, 1]
                den += w
            if den > 0:
                smooth_x[j] = num_x / den
                smooth_y[j] = num_y / den
    
    return smooth_x, smooth_y

# Points from the image approximation
center = [0, 0]
colored_points = [
    [ -0.4, -0.2 ],  # Purple
    [ -0.3, -0.3 ],  # Indigo
    [ 0.4, -0.3 ],   # Blue
    [ 0.5, 0.1 ],    # Cyan
    [ 0.3, 0.3 ],    # Green
    [ -0.2, 0.2 ]    # Yellow
]
colors = ['purple', 'indigo', 'blue', 'cyan', 'green', 'yellow']  # ROYGBIV minus red (center)

# 3D extension: Extrude along Z with height
height = 0.5
num_layers = 50  # For smooth 3D surface
z_levels = np.linspace(0, height, num_layers)

# Generate 3D points using green curve for each arm
arms_3d = []
for point in colored_points:
    # Points for arm: center to colored point
    arm_points = [center, point]
    kappas = [1.0, 1.0]  # Default weights
    smooth_x, smooth_y = custom_interoperations_green_curve(arm_points, kappas, is_closed=False)  # Open curve for arm
    # Extrude along Z
    arm_3d = np.zeros((len(smooth_x), 3))
    arm_3d[:, 0] = smooth_x
    arm_3d[:, 1] = smooth_y
    arms_3d.append(arm_3d)

# Create mesh for 3D ribit structure (simple extrusion for demonstration)
# For each arm, create a cylindrical extrusion or lofted surface; here, simple points for plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, arm_3d in enumerate(arms_3d):
    # Extrude arm along Z
    for z in z_levels:
        arm_z = arm_3d.copy()
        arm_z[:, 2] = z
        ax.plot(arm_z[:, 0], arm_z[:, 1], arm_z[:, 2], color=colors[i])
    # Colored point at end
    ax.scatter(arm_3d[-1, 0], arm_3d[-1, 1], height, color=colors[i], s=50)

# Center point
ax.scatter(center[0], center[1], height, color='red', s=100)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Ribit Structure with Green Curves')

plt.show()

print("The plot shows a black curve connecting three red points, with a gray parallel curve above it.")
