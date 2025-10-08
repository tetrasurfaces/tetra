# ribit_structure.py
# Copyright 2025 Beau Ayres
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
# This software is proprietary and confidential. Unauthorized copying,
# distribution, modification, or use is strictly prohibited without
# express written permission from Beau Ayres.
#
# AGPL-3.0-or-later licensed
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import hashlib
import math
import mpmath
mpmath.mp.dps = 19
from wise_transforms import hashwise_transform, hexwise_transform
from kappawise import murmur32, kappa_coord
from wise_transforms import bitwise_transform, hexwise_transform, hashwise_transform
from id_util_nurks_closure_hex import custom_interoperations_green_curve, bspline_basis

# Points from the image approximation (adjusted for symmetry, with one diameter offset for "not in line")
center = [0, 0]
colored_points = [
    [ -0.4, -0.2 ], # Orange
    [ -0.3, -0.3 ], # Yellow
    [ 0.4, -0.3 ], # Green
    [ 0.5, 0.1 ], # Blue
    [ 0.3, 0.3 ], # Indigo
    [ -0.2, 0.2 ] # Violet
]
colors = ['orange', 'yellow', 'green', 'blue', 'indigo', 'violet'] # ROYGBIV minus red (center red)

# 3D extension: Extrude along Z with height
height = 0.5
num_layers = 50 # For smooth 3D surface
z_levels = np.linspace(0, height, num_layers)

# Generate 3D points using green curve for each arm, with intermediate points for curvature
arms_3d = []
for point in colored_points:
    # Add intermediate point for curvature (midway, offset slightly for curved effect)
    mid_point = [ (center[0] + point[0]) / 2 + 0.05 * np.random.randn(), (center[1] + point[1]) / 2 + 0.05 * np.random.randn() ]
    arm_points = [center, mid_point, point]
    kappas = [1.0, 1.0, 1.0]  # Default weights
    smooth_x, smooth_y = custom_interoperations_green_curve(arm_points, kappas, is_closed=False)  # Open curve for arm
    # Extrude along Z
    arm_3d = np.zeros((len(smooth_x), 3))
    arm_3d[:, 0] = smooth_x
    arm_3d[:, 1] = smooth_y
    arms_3d.append(arm_3d)

# Create mesh for 3D ribit structure (simple extrusion for demonstration)
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

print("The 3D ribit structure is generated with smoothed green curves for each arm, extruded along Z for depth. The center is at (0,0) with colored points at petal ends.")
