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
from matplotlib.widgets import Slider
from matplotlib import cm

def bspline_basis(u, i, p, knots):
    """
    Recursive B-spline basis function.
    """
    if p == 0:
        return 1.0 if knots[i] <= u < knots[i + 1] else 0.0
    den1 = knots[i + p] - knots[i]
    den2 = knots[i + p + 1] - knots[i + 1]
    term1 = ((u - knots[i]) / den1 * bspline_basis(u, i, p - 1, knots)) if den1 > 0 else 0.0
    term2 = ((knots[i + p + 1] - u) / den2 * bspline_basis(u, i + 1, p - 1, knots)) if den2 > 0 else 0.0
    return term1 + term2

def compute_nurks_surface(ns_diameter=2.0, nw_se_diameter=1.5, ne_sw_diameter=1.8, twist=0.0, amplitude=0.3, radii=1.0, kappa=1.0, height=2.0, inflection=0.5, degree=3, res=50):
    """
    Compute the NURKS surface points for a 6-petal flower shape as a single surface body.
    
    Parameters:
    - ns_diameter, nw_se_diameter, ne_sw_diameter: Diameters for elliptical deformation of the boundary profile.
    - twist: Twist angle for petals.
    - amplitude: Amplitude of petal undulations (positive or negative for inflection).
    - radii: Base radius for the boundary profile.
    - kappa: Curvature modulation for Z.
    - height: Maximum height of the surface.
    - inflection: Controls the inflection point in the Z profile (0 to 1, where inflection occurs).
    - degree: Spline degree.
    - res: Resolution for u and v.
    
    Returns:
    - X, Y, Z: Meshgrid arrays for the surface.
    - control_x, control_y, control_z: Flattened control points for plotting.
    """
    num_petals = 6
    num_u = num_petals * 2  # 12 control points in angular direction
    num_v = degree + 1  # Radial control layers
    
    control_points = np.zeros((num_u, num_v, 3))
    weights = np.ones((num_u, num_v))
    
    a = ns_diameter / 2
    b = ne_sw_diameter / 2
    c = nw_se_diameter / 2
    
    for i in range(num_u):
        theta = (i / num_u * 2 * np.pi) + twist
        r_base = radii + amplitude * np.sin(num_petals * theta)
        x_base = r_base * np.cos(theta) * (a + c) / 2
        y_base = r_base * np.sin(theta) * (b + a) / 2
        for j in range(num_v):
            v = j / (num_v - 1) if num_v > 1 else 0
            scale = 1 - v  # Reverse scale: v=0 at boundary (flower profile), v=1 at center
            control_points[i, j, 0] = scale * x_base
            control_points[i, j, 1] = scale * y_base
            # Adjusted Z for inflection: sigmoid-like curve for better control over shape
            z_norm = 1 / (1 + np.exp(-kappa * (v - inflection) * 10))  # Steep transition at inflection point
            control_points[i, j, 2] = height * z_norm
    
    # For periodic in u: duplicate first degree rows
    control_points_u = np.concatenate((control_points, control_points[:degree, :, :]), axis=0)
    weights_u = np.concatenate((weights, weights[:degree, :]), axis=0)
    
    # Knot vectors (clamped)
    n_u = len(control_points_u) - 1
    knots_u = np.concatenate(([0] * (degree + 1), np.linspace(0, 1, n_u - degree + 2), [1] * (degree)))
    
    n_v = num_v - 1
    knots_v = np.concatenate(([0] * (degree + 1), np.linspace(0, 1, n_v - degree + 2), [1] * (degree)))
    
    # Evaluate surface
    u_vals = np.linspace(0, 1, res)
    v_vals = np.linspace(0, 1, res)
    U, V = np.meshgrid(u_vals, v_vals)
    
    X = np.zeros((res, res))
    Y = np.zeros((res, res))
    Z = np.zeros((res, res))
    
    for ii in range(res):
        for jj in range(res):
            u = u_vals[ii]
            v = v_vals[jj]
            denom = 0.0
            numer_x = numer_y = numer_z = 0.0
            for i in range(len(control_points_u)):
                N_i = bspline_basis(u, i, degree, knots_u)
                for j in range(num_v):
                    M_j = bspline_basis(v, j, degree, knots_v)
                    basis = N_i * M_j
                    w = weights_u[i, j] * basis
                    denom += w
                    numer_x += w * control_points_u[i, j, 0]
                    numer_y += w * control_points_u[i, j, 1]
                    numer_z += w * control_points_u[i, j, 2]
            if denom > 0:
                X[ii, jj] = numer_x / denom
                Y[ii, jj] = numer_y / denom
                Z[ii, jj] = numer_z / denom
    
    # Control points for plotting (original, without duplication)
    control_x = control_points[:, :, 0].flatten()
    control_y = control_points[:, :, 1].flatten()
    control_z = control_points[:, :, 2].flatten()
    
    return X, Y, Z, control_x, control_y, control_z

# Interactive visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.25)

# Slider axes
ax_ns = plt.axes([0.25, 0.22, 0.65, 0.03])
ax_nw = plt.axes([0.25, 0.17, 0.65, 0.03])
ax_ne = plt.axes([0.25, 0.12, 0.65, 0.03])
ax_twist = plt.axes([0.25, 0.07, 0.65, 0.03])
ax_amp = plt.axes([0.25, 0.02, 0.65, 0.03])
ax_radii = plt.axes([0.05, 0.25, 0.0225, 0.63], facecolor='lightgoldenrodyellow')
ax_kappa = plt.axes([0.10, 0.25, 0.0225, 0.63], facecolor='lightgoldenrodyellow')
ax_height = plt.axes([0.15, 0.25, 0.0225, 0.63], facecolor='lightgoldenrodyellow')
ax_inflection = plt.axes([0.20, 0.25, 0.0225, 0.63], facecolor='lightgoldenrodyellow')

# Initial values
init_ns = 2.0
init_nw = 1.5
init_ne = 1.8
init_twist = 0.2
init_amp = 0.3
init_radii = 1.0
init_kappa = 1.0
init_height = 2.0
init_inflection = 0.5

# Sliders
s_ns = Slider(ax_ns, 'NS Diam', 0.5, 3.0, valinit=init_ns)
s_nw = Slider(ax_nw, 'NW/SE Diam', 0.5, 3.0, valinit=init_nw)
s_ne = Slider(ax_ne, 'NE/SW Diam', 0.5, 3.0, valinit=init_ne)
s_twist = Slider(ax_twist, 'Twist', 0.0, np.pi, valinit=init_twist)
s_amp = Slider(ax_amp, 'Amplitude', -0.5, 0.5, valinit=init_amp)
s_radii = Slider(ax_radii, 'Radii', 0.1, 2.0, valinit=init_radii, orientation='vertical')
s_kappa = Slider(ax_kappa, 'Kappa', 0.5, 1.5, valinit=init_kappa, orientation='vertical')
s_height = Slider(ax_height, 'Height', 0.5, 3.0, valinit=init_height, orientation='vertical')
s_inflection = Slider(ax_inflection, 'Inflection', 0.0, 1.0, valinit=init_inflection, orientation='vertical')

def update(val):
    ns = s_ns.val
    nw = s_nw.val
    ne = s_ne.val
    twist = s_twist.val
    amp = s_amp.val
    radii = s_radii.val
    kappa = s_kappa.val
    height = s_height.val
    inflection = s_inflection.val
    
    X, Y, Z, control_x, control_y, control_z = compute_nurks_surface(ns, nw, ne, twist, amp, radii, kappa, height, inflection)
    
    ax.clear()
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.plot_wireframe(X, Y, Z, color='black', linewidth=0.5)
    
    # Show control points
    ax.scatter(control_x, control_y, control_z, color='red', s=50)
    
    # Control net lines (u and v directions)
    num_u = 12
    num_v = 4
    for j in range(num_v):
        cx = control_x[j::num_v]
        cy = control_y[j::num_v]
        cz = control_z[j::num_v]
        ax.plot(np.append(cx, cx[0]), np.append(cy, cy[0]), np.append(cz, cz[0]), 'k--')
    for i in range(num_u):
        cx = control_x[i * num_v : (i + 1) * num_v]
        cy = control_y[i * num_v : (i + 1) * num_v]
        cz = control_z[i * num_v : (i + 1) * num_v]
        ax.plot(cx, cy, cz, 'k--')
    
    ax.set_title('6-Petal NURKS Single Surface with Flower Profile Boundary')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    limits = max(np.max(np.abs(X)), np.max(np.abs(Y)), np.max(Z)) * 1.1
    ax.set_xlim(-limits, limits)
    ax.set_ylim(-limits, limits)
    ax.set_zlim(0, limits * 2)
    fig.canvas.draw_idle()

# Attach updates
s_ns.on_changed(update)
s_nw.on_changed(update)
s_ne.on_changed(update)
s_twist.on_changed(update)
s_amp.on_changed(update)
s_radii.on_changed(update)
s_kappa.on_changed(update)
s_height.on_changed(update)
s_inflection.on_changed(update)

# Initial update
update(None)

plt.show()
