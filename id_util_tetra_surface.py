# id_util_tetra_surface.py
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

# Define control points for a simple NURKS surface (e.g., a curved dome)
control_points = np.array([
    [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
    [[0, 1, 0], [1, 1, 2], [2, 1, 0]],
    [[0, 2, 0], [1, 2, 0], [2, 2, 0]]
])

# Weights for rational aspect - expand to match shape
weights = np.array([
    [1, 1, 1],
    [1, 1.5, 1],
    [1, 1, 1]
])

# Kappa parameter for curvature adjustment
kappa = 1.0  # Adjust this for more/less curvature

# Simple NURBS surface evaluation (bilinear for demo)
u = np.linspace(0, 1, 50)
v = np.linspace(0, 1, 50)
U, V = np.meshgrid(u, v)

X = np.zeros_like(U)
Y = np.zeros_like(U)
Z = np.zeros_like(U)

for i in range(len(u)):
    for j in range(len(v)):
        denom = 0
        numer_x, numer_y, numer_z = 0, 0, 0
        for a in range(3):
            for b in range(3):
                basis = (1 - u[i])**(2-a) * u[i]**a * (1 - v[j])**(2-b) * v[j]**b  # Bilinear basis (degree 2 approx)
                w = weights[a, b] * basis
                denom += w
                numer_x += w * control_points[a, b, 0]
                numer_y += w * control_points[a, b, 1]
                numer_z += w * control_points[a, b, 2] * kappa  # Kappa modulates Z
        X[i, j] = numer_x / denom
        Y[i, j] = numer_y / denom
        Z[i, j] = numer_z / denom

# Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z, color='black')
ax.plot_surface(X, Y, Z, cmap='viridis')

# Show control points
for a in range(3):
    for b in range(3):
        point = control_points[a, b]
        ax.scatter(point[0], point[1], point[2], color='red', s=50)

ax.set_title('NURKS Surface with Control Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
