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

def custom_interoperations_green_curve(points, kappas, is_closed=False):
    points = np.array(points)
    kappas = np.array(kappas)
    degree = 3
    num_output_points = 1000
   
    if is_closed and len(points) > degree:
        n = len(points)
        # Compute chordal parameterization
        t = [0.0]
        for i in range(n):
            j = (i + 1) % n
            dx = points[j, 0] - points[i, 0]
            dy = points[j, 1] - points[i, 1]
            t.append(t[-1] + np.sqrt(dx**2 + dy**2))
        t = np.array(t)
        t = t / t[-1]  # Normalize
        
        # Extend knots for periodicity
        knots_left = t[-degree:] - 1
        knots_right = t[:degree] + 1
        knots = np.concatenate((knots_left, t[:-1], knots_right))  # t[:-1] because t has n+1, but for n points
        
        extended_points = np.concatenate((points[-degree:], points, points[:degree]))
        extended_kappas = np.concatenate((kappas[-degree:], kappas, kappas[:degree]))
        len_extended = len(extended_points)
        
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
   
    else:
        # Original open case
        t = np.cumsum([0] + [np.linalg.norm(points[i+1] - points[i]) for i in range(len(points)-1)])
        knots = np.concatenate(([0] * (degree + 1), t / t[-1] if t[-1] > 0 else np.linspace(0, 1, len(t)), [1] * (degree)))
   
        u_fine = np.linspace(0, 1, num_output_points, endpoint=False)
   
        smooth_x = np.zeros(num_output_points)
        smooth_y = np.zeros(num_output_points)
   
        for j, u in enumerate(u_fine):
            num_x, num_y, den = 0.0, 0.0, 0.0
            for i in range(len(points)):
                b = bspline_basis(u, i, degree, knots)
                w = kappas[i] * b if i < len(kappas) else kappas[-1] * b
                num_x += w * points[i, 0]
                num_y += w * points[i, 1]
                den += w
            if den > 0:
                smooth_x[j] = num_x / den
                smooth_y[j] = num_y / den
   
    return smooth_x, smooth_y

# Test with flower shape
num_petals = 6
num_u = num_petals * 2  # 12
theta = np.linspace(0, 2 * np.pi, num_u, endpoint=False)
radii = 1.0
amplitude = 0.3
r_base = radii + amplitude * np.sin(num_petals * theta)
x_base_coarse = r_base * np.cos(theta)
y_base_coarse = r_base * np.sin(theta)

boundary_points = list(zip(x_base_coarse, y_base_coarse))
boundary_kappas = [1.0] * len(boundary_points)
smooth_x, smooth_y = custom_interoperations_green_curve(boundary_points, boundary_kappas, is_closed=True)

# Check if closed
print('First and last points match:', np.allclose(smooth_x[0], smooth_x[-1]), np.allclose(smooth_y[0], smooth_y[-1]))

# Plot to visualize
plt.figure()
plt.plot(smooth_x, smooth_y, 'b-')
plt.plot(x_base_coarse, y_base_coarse, 'r--')
plt.scatter(x_base_coarse, y_base_coarse, color='red')
plt.axis('equal')
plt.title('Smoothed Closed Curve with chordal knots')
plt.show()

# Output some points to check
print('Smooth X first 5:', smooth_x[:5])
print('Smooth Y first 5:', smooth_y[:5])
print('Smooth X last 5:', smooth_x[-5:])
print('Smooth Y last 5:', smooth_y[-5:])
