# id_util_tetra_profile.py
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def generate_flower_profile(ns_diameter=1.0, nw_se_diameter=0.8, ne_sw_diameter=0.9, twist=0.0, amplitude=0.2, radii=0.5, num_petals=7, num_points=1000):
    """
    Generate a flower-shaped top profile using parametric equations.
    
    Parameters:
    - ns_diameter: North-South diameter (vertical axis).
    - nw_se_diameter: Northwest-Southeast diameter (one diagonal).
    - ne_sw_diameter: Northeast-Southwest diameter (other diagonal).
    - twist: Twist angle in radians for petal rotation.
    - amplitude: Amplitude of petal undulations.
    - radii: Base radius scaling for petals.
    - num_petals: Number of petals in the flower shape.
    - num_points: Number of points to generate for the curve.
    
    Returns:
    - x, y: Arrays of x and y coordinates for the 2D flower profile.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    
    # Base elliptical shape using the diameters
    a = ns_diameter / 2  # Semi-major axis (NS)
    b = ne_sw_diameter / 2  # Semi-minor axis approximation
    c = nw_se_diameter / 2  # Another diagonal for variation
    
    # Parametric radius with petal undulations and twist
    r = radii + amplitude * np.sin(num_petals * (theta + twist))
    
    # Incorporate elliptical deformation
    x = r * np.cos(theta) * (a + c) / 2
    y = r * np.sin(theta) * (b + a) / 2
    
    # Apply twist to the entire shape
    twist_mat = np.array([[np.cos(twist), -np.sin(twist)],
                          [np.sin(twist), np.cos(twist)]])
    points = np.vstack([x, y])
    twisted_points = twist_mat @ points
    
    return twisted_points[0], twisted_points[1]

def bspline_basis(u, i, p, knots):
    """
    Recursive B-spline basis function.
    
    Parameters:
    - u: Parameter value.
    - i: Basis index.
    - p: Degree.
    - knots: Knot vector.
    
    Returns:
    - Basis value.
    """
    if p == 0:
        return 1.0 if knots[i] <= u < knots[i + 1] else 0.0
    den1 = knots[i + p] - knots[i]
    den2 = knots[i + p + 1] - knots[i + 1]
    term1 = ((u - knots[i]) / den1 * bspline_basis(u, i, p - 1, knots)) if den1 > 0 else 0.0
    term2 = ((knots[i + p + 1] - u) / den2 * bspline_basis(u, i + 1, p - 1, knots)) if den2 > 0 else 0.0
    return term1 + term2

def apply_kspline_smoothing(x, y, kappa=1.0, degree=5, num_output_points=1000, num_controls=28):
    """
    Apply a kappa-modulated spline smoothing (KSpline) to the profile points using pure NumPy.
    This is a Non-Uniform Rational Kappa Spline (NURKS) implementation without SciPy dependencies.
    Subsamples input points to control points, uses constant weights based on kappa for rational spline.
    
    Parameters:
    - x, y: Input coordinates.
    - kappa: Curvature modulation factor (used as uniform weight).
    - degree: Spline degree.
    - num_output_points: Number of points in the output curve.
    - num_controls: Number of control points to subsample.
    
    Returns:
    - smooth_x, smooth_y: Smoothed coordinates.
    """
    # Subsample to control points
    idx = np.linspace(0, len(x) - 1, num_controls, dtype=int)
    control_points = np.array([[x[i], y[i]] for i in idx])
    
    # Weights uniform based on kappa
    weights = np.full(len(control_points), kappa)
    
    # For closed curve, append points for periodicity
    control_points = np.concatenate((control_points, control_points[1:degree + 1]))
    weights = np.concatenate((weights, weights[1:degree + 1]))
    
    n = len(control_points) - 1
    # Clamped uniform knot vector
    knots = np.concatenate(([0] * (degree + 1), np.linspace(0, 1, n - degree + 2), [1] * (degree)))
    
    u_fine = np.linspace(0, 1, num_output_points)
    
    smooth_x = np.zeros(num_output_points)
    smooth_y = np.zeros(num_output_points)
    
    for j, u in enumerate(u_fine):
        num_x, num_y, den = 0.0, 0.0, 0.0
        for i in range(len(control_points)):
            b = bspline_basis(u, i, degree, knots)
            w = weights[i] * b
            num_x += w * control_points[i, 0]
            num_y += w * control_points[i, 1]
            den += w
        if den > 0:
            smooth_x[j] = num_x / den
            smooth_y[j] = num_y / den
    
    return smooth_x, smooth_y

def generate_tetra_surface(x_profile, y_profile, height=1.0, num_layers=10, twist_per_layer=0.1):
    """
    Generate a 3D tetrahedral-like surface by extruding the 2D profile.
    This creates a stacked mesh with twisting layers for a petal-shaped volume.
    
    Parameters:
    - x_profile, y_profile: 2D profile coordinates.
    - height: Total height of the 3D mesh.
    - num_layers: Number of layers for extrusion.
    - twist_per_layer: Twist increment per layer.
    
    Returns:
    - vertices: 3D vertices (Nx3 array).
    - faces: Triangular faces (Mx3 array).
    """
    num_points = len(x_profile)
    vertices = []
    faces = []
    
    dz = height / num_layers
    for layer in range(num_layers + 1):
        z = layer * dz
        twist = layer * twist_per_layer
        rot_mat = np.array([[np.cos(twist), -np.sin(twist)],
                            [np.sin(twist), np.cos(twist)]])
        rotated = rot_mat @ np.vstack([x_profile, y_profile])
        for i in range(num_points):
            vertices.append([rotated[0, i], rotated[1, i], z])
    
    # Create triangular faces (simple extrusion triangulation)
    for layer in range(num_layers):
        base_idx = layer * num_points
        next_idx = (layer + 1) * num_points
        for i in range(num_points):
            j = (i + 1) % num_points
            # Two triangles per quad
            faces.append([base_idx + i, base_idx + j, next_idx + i])
            faces.append([base_idx + j, next_idx + j, next_idx + i])
    
    return np.array(vertices), np.array(faces)

def visualize_mesh(vertices, faces):
    """
    Visualize the 3D mesh using matplotlib.
    
    Parameters:
    - vertices: 3D vertices.
    - faces: Triangular faces.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(vertices[faces], alpha=0.5, edgecolor='k'))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    limits = np.ptp(vertices, axis=0) / 2
    ax.set_xlim(-limits[0], limits[0])
    ax.set_ylim(-limits[1], limits[1])
    ax.set_zlim(0, limits[2] * 2)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate flower profile
    x, y = generate_flower_profile(ns_diameter=2.0, nw_se_diameter=1.5, ne_sw_diameter=1.8, twist=0.2, amplitude=0.3, radii=1.0, num_petals=7)
    
    # Smooth with KSpline (pure NumPy implementation)
    smooth_x, smooth_y = apply_kspline_smoothing(x, y, kappa=0.9)
    
    # Generate tetra surface
    vertices, faces = generate_tetra_surface(smooth_x, smooth_y, height=2.0, num_layers=20, twist_per_layer=0.05)
    
    # Visualize
    visualize_mesh(vertices, faces)
