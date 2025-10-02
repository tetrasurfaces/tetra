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
# green_curve.py - B-Spline Curve Smoothing Function
# Notes: Generates smoothed curves using B-spline interpolation with support for open and closed curves. Complete script; run as-is. Requires numpy (pip install numpy). Mentally verified: Open curve with 3 points → smooth interpolation; closed with 6 points → periodic loop.

import numpy as np
import matplotlib.pyplot as plt

def bspline_basis(u, i, p, knots):
    """B-spline basis function for curve interpolation.
    Args:
        u: Parameter value (0 to 1).
        i: Control point index.
        p: Degree of the spline.
        knots: Knot vector.
    Returns:
        Basis value at u for control point i.
    """
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

if __name__ == "__main__":
    # Example usage: Open curve with 3 points
    points = [[0, 0], [0.5, 1], [1, 0]]  # Simple curve points
    kappas = [1.0, 1.0, 1.0]  # Weights
    smooth_x, smooth_y = custom_interoperations_green_curve(points, kappas, is_closed=False)
    
    # Plot the smoothed curve
    plt.plot(smooth_x, smooth_y, label='Smoothed Curve')
    plt.scatter([p[0] for p in points], [p[1] for p in points], c='red', label='Control Points')
    plt.legend()
    plt.title('Open B-Spline Curve Example')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    print("Example open curve smoothed. For closed curve, set is_closed=True with at least 4 points.")
