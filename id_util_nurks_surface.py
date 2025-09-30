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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider
from matplotlib import cm

def bspline_basis(u, i, p, knots):
    """
    Recursive B-spline basis function with index bounds checks to prevent out-of-range errors.
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

def bspline_basis_periodic(u, i, p, knots, n):
    """
    Periodic version of the B-spline basis function for closed curves.
    Wraps indices and adjusts denominators for periodicity.
    """
    i = i % n
    if p == 0:
        k0 = knots[i % len(knots)]
        k1 = knots[(i + 1) % len(knots)]
        if k0 > k1:  # Wrap-around interval
            return 1.0 if u >= k0 or u < k1 else 0.0
        else:
            return 1.0 if k0 <= u < k1 else 0.0
    
    k_i = knots[i % len(knots)]
    k_ip = knots[(i + p) % len(knots)]
    den1 = k_ip - k_i
    if den1 < 0:
        den1 += 1.0  # Adjust for wrap-around
    term1 = 0.0
    if den1 > 0:
        term1 = ((u - k_i) / den1) * bspline_basis_periodic(u, i, p - 1, knots, n)
    
    k_i1 = knots[(i + 1) % len(knots)]
    k_ip1 = knots[(i + p + 1) % len(knots)]
    den2 = k_ip1 - k_i1
    if den2 < 0:
        den2 += 1.0  # Adjust for wrap-around
    term2 = 0.0
    if den2 > 0:
        term2 = ((k_ip1 - u) / den2) * bspline_basis_periodic(u, i + 1, p - 1, knots, n)
    
    return term1 + term2

def custom_interoperations_green_curve(points, kappas, is_closed=False):
    """
    Custom Non-Uniform Rational Kappa Spline (NURKS) approximation for green curve with closure adjustments.
    Uses periodic B-spline basis for closed curves to ensure smooth closure without external libraries.
    """
    points = np.array(points)
    kappas = np.array(kappas)
    degree = 3  # Fixed degree for continuity
    num_output_points = 1000
    
    if is_closed:
        n = len(points)
        knots = np.linspace(0, 1, n + 1)
        u_fine = np.linspace(0, 1, num_output_points, endpoint=False)
        
        smooth_x = np.zeros(num_output_points)
        smooth_y = np.zeros(num_output_points)
        
        for j, u in enumerate(u_fine):
            num_x, num_y, den = 0.0, 0.0, 0.0
            for i in range(n):
                b = bspline_basis_periodic(u, i, degree, knots, n)
                w = kappas[i] * b
                num_x += w * points[i, 0]
                num_y += w * points[i, 1]
                den += w
            if den > 0:
                smooth_x[j] = num_x / den
                smooth_y[j] = num_y / den
        
        smooth_x = np.append(smooth_x, smooth_x[0])
        smooth_y = np.append(smooth_y, smooth_y[0])
    else:
        # Non-closed case using standard B-spline
        if len(points) > degree:
            points = np.concatenate((points, points[0:degree]))
            kappas = np.concatenate((kappas, kappas[0:degree]))
        
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

def compute_nurks_surface(ns_diameter=2.0, nw_se_diameter=1.5, ne_sw_diameter=1.8, twist=0.0, amplitude=0.3, inner_amp=0.3, radii=1.0, kappa=1.0, height=2.0, inflection=0.5, radial_bend=0.0, inner_radius=0.01, degree=3, res=50):
    """
    Compute the NURKS surface points for a 6-petal flower shape as a single surface body.
   
    Parameters:
    - ns_diameter, nw_se_diameter, ne_sw_diameter: Diameters for elliptical deformation of the boundary profile.
    - twist: Twist angle for petals.
    - amplitude: Amplitude of petal undulations for outer (positive or negative for inflection).
    - inner_amp: Amplitude for inner (centre) flower profile.
    - radii: Base radius for the boundary profile.
    - kappa: Curvature modulation for Z.
    - height: Maximum height of the surface.
    - inflection: Controls the inflection point in the Z profile (0 to 1, where inflection occurs).
    - radial_bend: Bend factor for curving radial lines.
    - inner_radius: Small inner radius to avoid fan artefact.
    - degree: Spline degree.
    - res: Resolution for u and v.
   
    Returns:
    - vertices: 3D points (Nx3).
    - faces: Triangular faces (Mx3).
    - face_colors: Colors for faces.
    - control_x, control_y, control_z: Flattened control points for plotting.
    - res: Resolution used.
    """
    num_petals = 6
    num_u = num_petals * 2  # 12 control points in angular direction
    num_v = degree + 1  # Radial control layers
   
    control_points = np.zeros((num_u, num_v, 3))
    weights = np.ones((num_u, num_v))
   
    a = ns_diameter / 2
    b = ne_sw_diameter / 2
    c = nw_se_diameter / 2
   
    theta = np.linspace(0, 2 * np.pi, num_u, endpoint=False) + twist
   
    # Generate coarse boundary points
    r_base = radii + amplitude * np.sin(num_petals * theta)
    x_base_coarse = r_base * np.cos(theta) * (a + c) / 2
    y_base_coarse = r_base * np.sin(theta) * (b + a) / 2
   
    # Use custom green curve to smooth the boundary profile with closure
    boundary_points = list(zip(x_base_coarse, y_base_coarse))
    boundary_kappas = [1.0] * len(boundary_points)
    smooth_x, smooth_y = custom_interoperations_green_curve(boundary_points, boundary_kappas, is_closed=True)
   
    # Resample smooth boundary to num_u for control, avoiding duplicate last point for distinct controls
    idx = np.linspace(0, len(smooth_x) - 2, num_u, dtype=int)
    x_base = smooth_x[idx]
    y_base = smooth_y[idx]
   
    # Precompute max for Z normalization across all radial layers
    raw_z = np.zeros(num_v)
    for j in range(num_v):
        v = j / (num_v - 1) if num_v > 1 else 0
        dist = abs(v - inflection)
        raw_z[j] = dist ** kappa
    max_raw = np.max(raw_z) if np.max(raw_z) > 0 else 1e-10
   
    for i in range(num_u):
        for j in range(num_v):
            v = j / (num_v - 1) if num_v > 1 else 0
            if j == 0:
                # Use smoothed boundary for outer profile (v=0)
                x_v = x_base[i]
                y_v = y_base[i]
            else:
                # Compute inner profiles using parametric formula
                theta_v = theta[i] + radial_bend * np.sin(np.pi * v)  # Sin for V-like bend in radial
                amp_v = amplitude * (1 - v) + inner_amp * v  # Blend outer to inner amplitude
                r_v = radii + amp_v * np.sin(num_petals * theta_v)
                x_v = r_v * np.cos(theta_v) * (a + c) / 2
                y_v = r_v * np.sin(theta_v) * (b + a) / 2
            scale = inner_radius + (1 - inner_radius) * (1 - v)
            control_points[i, j, 0] = scale * x_v
            control_points[i, j, 1] = scale * y_v
            # Curved V-angulation radial profile: point down with curved arms
            dist = abs(v - inflection)
            z_norm = (dist ** kappa)
            z_norm = 1 - z_norm / max_raw
            control_points[i, j, 2] = height * z_norm
   
    # For periodic in u: duplicate first degree rows
    control_points_u = np.concatenate((control_points, control_points[:degree, :, :]), axis=0)
    weights_u = np.concatenate((weights, weights[:degree, :]), axis=0)
   
    # Clamped knot vector for periodic u
    n_u = len(control_points_u) - 1
    knots_u = np.concatenate(([0] * (degree + 1), np.linspace(0, 1, n_u - degree + 2), [1] * (degree)))
   
    # Clamped for v
    n_v = num_v - 1
    knots_v = np.concatenate(([0] * (degree + 1), np.linspace(0, 1, n_v - degree + 2), [1] * (degree)))
   
    # Evaluate surface points
    u_vals = np.linspace(0, 1, res, endpoint=False)
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
   
    # Make seamless by setting last to first for periodic
    X[:, -1] = X[:, 0]
    Y[:, -1] = Y[:, 0]
    Z[:, -1] = Z[:, 0]
   
    # Flatten to vertices
    vertices = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
   
    # Create Delaunay-like triangular faces by splitting grid quads
    faces = []
    for i in range(res - 1):
        for j in range(res):
            base = i * res + j
            next_base = (i + 1) * res + j
            j1 = (j + 1) % res
            base1 = i * res + j1
            next_base1 = (i + 1) * res + j1
            faces.append([base, base1, next_base])
            faces.append([base1, next_base1, next_base])
    faces = np.array(faces)
   
    # Face colors based on average Z
    face_colors = []
    for f in faces:
        avg_z = np.mean(vertices[f, 2])
        norm = avg_z / height if height > 0 else 0
        face_colors.append(cm.viridis(norm))
   
    # Control points for plotting (original, without duplication)
    control_x = control_points[:, :, 0].flatten()
    control_y = control_points[:, :, 1].flatten()
    control_z = control_points[:, :, 2].flatten()
   
    return vertices, faces, face_colors, control_x, control_y, control_z, res

# Interactive visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.25)
# Slider axes
ax_ns = plt.axes([0.25, 0.32, 0.65, 0.03])
ax_nw = plt.axes([0.25, 0.27, 0.65, 0.03])
ax_ne = plt.axes([0.25, 0.22, 0.65, 0.03])
ax_twist = plt.axes([0.25, 0.17, 0.65, 0.03])
ax_amp = plt.axes([0.25, 0.12, 0.65, 0.03])
ax_inner_amp = plt.axes([0.25, 0.07, 0.65, 0.03])
ax_radii = plt.axes([0.05, 0.25, 0.0225, 0.63], facecolor='lightgoldenrodyellow')
ax_kappa = plt.axes([0.10, 0.25, 0.0225, 0.63], facecolor='lightgoldenrodyellow')
ax_height = plt.axes([0.15, 0.25, 0.0225, 0.63], facecolor='lightgoldenrodyellow')
ax_inflection = plt.axes([0.20, 0.25, 0.0225, 0.63], facecolor='lightgoldenrodyellow')
ax_bend = plt.axes([0.25, 0.02, 0.65, 0.03])
# Initial values
init_ns = 2.0
init_nw = 1.5
init_ne = 1.8
init_twist = 0.2
init_amp = 0.3
init_inner_amp = 0.3
init_radii = 1.0
init_kappa = 1.0
init_height = 2.0
init_inflection = 0.5
init_bend = 0.5
# Sliders
s_ns = Slider(ax_ns, 'NS Diam', 0.5, 3.0, valinit=init_ns)
s_nw = Slider(ax_nw, 'NW/SE Diam', 0.5, 3.0, valinit=init_nw)
s_ne = Slider(ax_ne, 'NE/SW Diam', 0.5, 3.0, valinit=init_ne)
s_twist = Slider(ax_twist, 'Twist', 0.0, np.pi, valinit=init_twist)
s_amp = Slider(ax_amp, 'Amplitude', -0.5, 0.5, valinit=init_amp)
s_inner_amp = Slider(ax_inner_amp, 'Inner Amp', -0.5, 0.5, valinit=init_inner_amp)
s_radii = Slider(ax_radii, 'Radii', 0.1, 2.0, valinit=init_radii, orientation='vertical')
s_kappa = Slider(ax_kappa, 'Kappa', 0.5, 1.5, valinit=init_kappa, orientation='vertical')
s_height = Slider(ax_height, 'Height', 0.5, 3.0, valinit=init_height, orientation='vertical')
s_inflection = Slider(ax_inflection, 'Inflection', 0.0, 1.0, valinit=init_inflection, orientation='vertical')
s_bend = Slider(ax_bend, 'Radial Bend', 0.0, np.pi, valinit=init_bend)

def update(val):
    ns = s_ns.val
    nw = s_nw.val
    ne = s_ne.val
    twist = s_twist.val
    amp = s_amp.val
    inner_amp = s_inner_amp.val
    radii = s_radii.val
    kappa = s_kappa.val
    height = s_height.val
    inflection = s_inflection.val
    radial_bend = s_bend.val
   
    vertices, faces, face_colors, control_x, control_y, control_z, res = compute_nurks_surface(ns, nw, ne, twist, amp, inner_amp, radii, kappa, height, inflection, radial_bend)
   
    ax.clear()
    ax.add_collection3d(Poly3DCollection(vertices[faces], facecolors=face_colors, edgecolor='none', alpha=0.8))
    ax.plot_wireframe(vertices[:,0].reshape(res, res), vertices[:,1].reshape(res, res), vertices[:,2].reshape(res, res), color='black', linewidth=0.5)
   
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
    limits = max(np.max(np.abs(vertices[:,0])), np.max(np.abs(vertices[:,1])), np.max(vertices[:,2])) * 1.1
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
s_inner_amp.on_changed(update)
s_radii.on_changed(update)
s_kappa.on_changed(update)
s_height.on_changed(update)
s_inflection.on_changed(update)
s_bend.on_changed(update)
# Initial update
update(None)
plt.show()
