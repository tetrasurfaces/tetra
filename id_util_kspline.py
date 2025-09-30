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

# Points from the image approximation
points = np.array([[0.1, 0.1], [0.5, 0.3], [0.9, 0.2]])

# Simple curve (black)
t = np.linspace(0, 1, 100)
x_curve = (1 - t)**2 * points[0, 0] + 2*(1 - t)*t * points[1, 0] + t**2 * points[2, 0]
y_curve = (1 - t)**2 * points[0, 1] + 2*(1 - t)*t * points[1, 1] + t**2 * points[2, 1]

# Gray offset curve (parallel above)
offset = 0.05  # approximate offset
normals_x = np.gradient(y_curve)  # perpendicular is gradient of y for x, but approximate
normals_y = -np.gradient(x_curve)
norm = np.sqrt(normals_x**2 + normals_y**2)
normals_x /= norm
normals_y /= norm
x_offset = x_curve + offset * normals_x
y_offset = y_curve + offset * normals_y

# Plot
fig, ax = plt.subplots()
ax.plot(x_curve, y_curve, 'k-')
ax.plot(x_offset, y_offset, 'gray')
ax.scatter(points[:,0], points[:,1], c='red', s=50)
ax.set_axis_off()
plt.show()

print("The plot shows a black curve connecting three red points, with a gray parallel curve above it.")
