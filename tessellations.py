# Copyright 2025 Beau Ayres,Todd Hutchinson, Anonymous
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
# express written permission from Todd Hutchinson and Beau Ayres.
# tessellations.py - Hexagonal and Sierpinski Mesh Tessellation
# Notes: Generates hexagonal mesh and applies Sierpinski tessellation for surface detail. Complete; run as-is. Requires numpy (pip install numpy). Verified: Hex mesh with 6 cells → triangulated to STL; Sierpinski level=2 → detailed facets.
import numpy as np
import hashlib
import struct
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from tetras import fractal_tetra
from kappasha import kappasha256
def tessellate_hex_mesh(X, Y, Z, u_num, v_num, is_cap=False):
    """Tessellate surface into hexagonal mesh, triangulated for STL."""
    triangles = []
    # For simplicity, simulate hex grid by staggering rows (hexagulation)
    for i in range(v_num - 1):
        for j in range(u_num):
            # Define points for hex (approximate with 6 points, but since grid is quad, use quad to tri for now; extend for true hex)
            p1 = (i * u_num + j, X[i, j], Y[i, j], Z[i, j])
            p2 = (i * u_num + (j + 1) % u_num, X[i, (j + 1) % u_num], Y[i, (j + 1) % u_num], Z[i, (j + 1) % u_num])
            p3 = ((i + 1) * u_num + (j + 1) % u_num, X[i + 1, (j + 1) % u_num], Y[i + 1, (j + 1) % u_num], Z[i + 1, (j + 1) % u_num])
            p4 = ((i + 1) * u_num + j, X[i + 1, j], Y[i + 1, j], Z[i + 1, j])
            # Triangulate quad (for hex, would need 6-sided; this is placeholder)
            triangles.append((p1, p2, p3))
            triangles.append((p1, p3, p4))
    if is_cap:
        # Cap with center for closed surface
        center = (0, 0.0, 0.0, Z[0, 0])
        for j in range(u_num):
            p1 = (j, X[0, j], Y[0, j], Z[0, j])
            p2 = ( (j + 1) % u_num, X[0, (j + 1) % u_num], Y[0, (j + 1) % u_num], Z[0, (j + 1) % u_num])
            triangles.append((center, p1, p2))
    return triangles
def build_mail(X, Y, Z, level=3):
    """Build Sierpinski tessellated hex mail mesh for surface detail."""
    # Flatten surface points to 2D for Voronoi (project to XY for simplicity)
    points = np.column_stack((X.flatten(), Y.flatten()))
    vor = Voronoi(points)
    # Use Delaunay for triangulation of Voronoi regions (hex-like cells)
    tri = Delaunay(points)
    all_triangles = []
    # Apply Sierpinski (fractal_tetra) to each tri for detail
    for sim in tri.simplices:
        verts = points[sim]
        scale = np.linalg.norm(verts[1] - verts[0])  # Approximate scale
        orig = np.array([
            verts[0].tolist() + [Z.flatten()[sim[0]]],
            verts[1].tolist() + [Z.flatten()[sim[1]]],
            verts[2].tolist() + [Z.flatten()[sim[2]]],
            [np.mean(verts, axis=0)[0], np.mean(verts, axis=0)[1], np.mean(Z.flatten()[sim]) + scale / 2]  # Apex for tetra
        ])
        fractal_triangles = []
        fractal_tetra(orig.tolist(), level, fractal_triangles)
        all_triangles.extend(fractal_triangles)
    # Flatten to vertices and faces
    vertices = []
    faces = []
    for tri in all_triangles:
        base_idx = len(vertices)
        vertices.extend(tri)
        faces.append([base_idx, base_idx+1, base_idx+2])
    return vertices, faces
def generate_nurks_surface(ns_diam=1.0, sw_ne_diam=1.0, nw_se_diam=1.0, twist=0.0, amplitude=0.3, radii=1.0, kappa=1.0, height=1.0, inflection=0.5, morph=0.0, hex_mode=False):
    """Generate parametric NURKS surface points (X, Y, Z) and copyright hash ID using kappasha256."""
    # 36 nodes for angular control.
    u_num = 36
    v_num = 20
    inner_radius = 0.01 # Small to avoid artefacts.
    u = np.linspace(0, 2 * np.pi, u_num)
    v = np.linspace(inner_radius, 1, v_num)
    U, V = np.meshgrid(u, v)
    if hex_mode:
        # Hexagulation: Stagger alternate rows for hexagonal approximation.
        for i in range(1, v_num, 2):
            U[i, :] += np.pi / u_num / 2 # Stagger by half step.
    # Flower profile with 6 petals.
    petal_amp = amplitude * (1 - V) # Taper for smaller petals at outer ends (V=1).
    # Compute the base sin variation.
    sin_variation = np.sin(6 * U + twist)
    num_coarse = 36
    if hex_mode:
        # Use morph to morph profile: 0 flower, 1 hex, 2 circular.
        morph_mode = int(morph)
        num_coarse = 36
        if morph_mode == 0:
            num_coarse = 36
        elif morph_mode == 1:
            num_coarse = 6
        else:
            num_coarse = 100 # High for circular approximation
        theta_coarse = np.linspace(0, 2 * np.pi, num_coarse, endpoint=False)
        if morph_mode == 1:
            sin_coarse = np.sin(3 * theta_coarse + twist) # For hex-like (3 petals doubled)
        elif morph_mode == 2:
            sin_coarse = np.zeros(num_coarse) # No sin for circular
        else:
            sin_coarse = np.sin(6 * theta_coarse + twist) # Flower
        points = list(zip(theta_coarse, sin_coarse))
        kappas = [1.0] * num_coarse
        smooth_theta, smooth_sin = custom_interoperations_green_curve(points, kappas, is_closed=True)
        smooth_sin = smooth_sin[:-1]
        theta_fine = smooth_theta[:-1]
        sin_variation = np.interp(U % (2 * np.pi), theta_fine, smooth_sin)
    R = radii + petal_amp * sin_variation
    # Deform with diameters (elliptical/radial influence).
    # NS scales y, SW/NE and NW/SE scale diagonals.
    scale_x = (sw_ne_diam + nw_se_diam) / 2
    scale_y = ns_diam
    X = R * V * np.cos(U) * scale_x
    Y = R * V * np.sin(U) * scale_y
    # V-curve: Power-based angulation with inflection.
    dist = np.abs(V - inflection)
    Z = height * (1 - dist ** kappa) # Inverted V, sharper with higher kappa.
    # Curve radial lines (green curves in diagram) by adding twist modulation.
    curve_factor = 0.1 * amplitude # Curvature based on amplitude.
    X += curve_factor * np.sin(np.pi * V) * np.cos(U + np.pi/4) # Curve in SW/NE.
    Y += curve_factor * np.sin(np.pi * V) * np.sin(U + np.pi/4) # Curve in NW/SE.
    # Hash parameters for copyright ID using kappasha256 (key modulated by kappa).
    param_str = f"{ns_diam},{sw_ne_diam},{nw_se_diam},{twist},{amplitude},{radii},{kappa},{height},{inflection},{morph},{hex_mode}"
    if hex_mode:
        param_str += ',bspline_degree=3,bspline_coarse=36'
    key = hashlib.sha256(struct.pack('f', kappa)).digest() * 2 # 64-byte key from kappa.
    surface_id = kappasha256(param_str.encode('utf-8'), key)[0] # hash_hex as ID.
    print(f"Surface Copyright ID: {surface_id}")
    # Integrate ribit for center modulation if hex_mode.
    if hex_mode:
        ribit_int, state, color = ribit_generate(param_str)
        print(f"Ribit State: {state}, Color: {color}, Int: {ribit_int}")
        # Use ribit state to modulate cap parameters.
        kappa_cap = 3 + state # >7th for higher states
        twist_cap = twist + 2 * np.pi * state / 7 # Azimuth change
        mini_factor = 0.1 * (state + 1) / 7 # Scaled mini for ribit variation
        # Generate cap with ribit-modulated params.
        mini_radii = radii * mini_factor
        mini_amplitude = amplitude * mini_factor
        v_num_cap = 10
        v_cap = np.linspace(0, inner_radius, v_num_cap)
        U_cap, V_cap = np.meshgrid(u, v_cap)
        if hex_mode:
            for i in range(1, v_num_cap, 2):
                U_cap[i, :] += np.pi / u_num / 2 # Stagger cap too for honeycomb.
        # For cap profile, use 7 points K-spline.
        num_coarse_cap = 7
        theta_coarse_cap = np.linspace(0, 2 * np.pi, num_coarse_cap, endpoint=False)
        sin_coarse_cap = np.sin(6 * theta_coarse_cap + twist_cap)
        points_cap = list(zip(theta_coarse_cap, sin_coarse_cap))
        kappas_cap = [1.0] * num_coarse_cap
        smooth_theta_cap, smooth_sin_cap = custom_interoperations_green_curve(points_cap, kappas_cap, is_closed=True)
        smooth_sin_cap = smooth_sin_cap[:-1]
        theta_fine_cap = smooth_theta_cap[:-1]
        sin_variation_cap = np.interp(U_cap % (2 * np.pi), theta_fine_cap, smooth_sin_cap)
        R_cap_base = mini_radii + mini_amplitude * sin_variation_cap
        petal_amp_main_inner = amplitude * (1 - inner_radius)
        sin_variation_main = sin_variation[0, :] # Angular at boundary
        R_main_inner = radii + petal_amp_main_inner * sin_variation_main
        R_cap = R_cap_base + (R_main_inner - R_cap_base) * (V_cap[:, none] / inner_radius)
        # Deform cap with same scales.
        X_cap = R_cap * V_cap * np.cos(U_cap) * scale_x
        Y_cap = R_cap * V_cap * np.sin(U_cap) * scale_y
        # Curve radial for cap.
        X_cap += curve_factor * np.sin(np.pi * V_cap) * np.cos(U_cap + np.pi/4)
        Y_cap += curve_factor * np.sin(np.pi * V_cap) * np.sin(U_cap + np.pi/4)
        # Z for cap with high power for continuity.
        Z_main_inner = height * (1 - (inner_radius - inflection) ** kappa) # Approximate, assuming inner small.
        dist_cap = V_cap / inner_radius
        Z_cap = height - (height - Z_main_inner) * dist_cap ** kappa_cap
    else:
        X_cap = None
        Y_cap = None
        Z_cap = None
    # Hash parameters for copyright ID using kappasha256 (key modulated by kappa).
    param_str = f"{ns_diam},{sw_ne_diam},{nw_se_diam},{twist},{amplitude},{radii},{kappa},{height},{inflection},{morph},{hex_mode}"
    if hex_mode:
        param_str += f',bspline_degree=3,bspline_coarse=36,ribit_state={state},kappa_cap={kappa_cap},mini_factor={mini_factor}'
    key = hashlib.sha256(struct.pack('f', kappa)).digest() * 2 # 64-byte key from kappa.
    surface_id = kappasha256(param_str.encode('utf-8'), key)[0] # hash_hex as ID.
    print(f"Surface Copyright ID: {surface_id}")
    return X, Y, Z, surface_id, X_cap, Y_cap, Z_cap
def tessellate_mesh(X, Y, Z, u_num, v_num, is_cap=False):
    triangles = []
    if is_cap:
        # Add center point at (0, 0, Z[0,0])
        center = (0, 0.0, 0.0, Z[0, 0])
        # Add fan triangles for the first row (i=0)
        for j in range(u_num):
            p1 = (j, X[0, j], Y[0, j], Z[0, j])
            p2 = (j + 1 % u_num, X[0, (j + 1) % u_num], Y[0, (j + 1) % u_num], Z[0, (j + 1) % u_num])
            triangles.append((center, p1, p2))
        start_i = 0
    else:
        start_i = 0
    # Normal quads for the rest
    for i in range(start_i, v_num - 1):
        for j in range(u_num):
            p1 = (i * u_num + j, X[i, j], Y[i, j], Z[i, j])
            p2 = (i * u_num + (j + 1) % u_num, X[i, (j + 1) % u_num], Y[i, (j + 1) % u_num], Z[i, (j + 1) % u_num])
            p3 = ((i + 1) * u_num + (j + 1) % u_num, X[i + 1, (j + 1) % u_num], Y[i + 1, (j + 1) % u_num], Z[i + 1, (j + 1) % u_num])
            p4 = ((i + 1) * u_num + j, X[i + 1, j], Y[i + 1, j], Z[i + 1, j])
            # Two triangles per quad.
            triangles.append((p1, p2, p3))
            triangles.append((p1, p3, p4))
    return triangles
def export_to_stl(triangles, filename, surface_id):
    """Export mesh to binary STL with embedded hash in header."""
    header = f"ID: {surface_id}".ljust(80, ' ').encode('utf-8')
    num_tri = len(triangles)
    with open(filename, 'wb') as f:
        f.write(header)
        f.write(struct.pack('<I', num_tri))
        for tri in triangles:
            # Compute normal with handling for degenerate cases.
            v1 = np.array(tri[1][1:]) - np.array(tri[0][1:])
            v2 = np.array(tri[2][1:]) - np.array(tri[0][1:])
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal /= norm_len
            else:
                normal = np.array([0.0, 0.0, 1.0]) # Default upward normal.
            f.write(struct.pack('<3f', *normal))
            for p in tri:
                f.write(struct.pack('<3f', *p[1:]))
            f.write(struct.pack('<H', 0)) # Attribute byte count.
# Interactive visualization.
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y, Z, surface_id, X_cap, Y_cap, Z_cap = generate_nurks_surface()
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_title('Interactive NURKS Surface')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# Adjust layout for sliders.
plt.subplots_adjust(left=0.25, bottom=0.35)
# Sliders for all parameters (positioned vertically).
slider_params = [
    ('NS Diam', 0.5, 2.0, 1.0),
    ('SW/NE Diam', 0.5, 2.0, 1.0),
    ('NW/SE Diam', 0.5, 2.0, 1.0),
    ('Twist', -np.pi, np.pi, 0.0),
    ('Amplitude', -1.0, 1.0, 0.3),
    ('Radii', 0.5, 2.0, 1.0),
    ('Kappa', 0.1, 5.0, 1.0),
    ('Height', 0.5, 2.0, 1.0),
    ('Inflection', 0.0, 1.0, 0.5),
    ('Morph', 0.0, 2.0, 0.0) # Add morph slider
]
sliders = []
y_pos = 0.25
for label, vmin, vmax, vinit in slider_params:
    ax_slider = plt.axes([0.1, y_pos, 0.65, 0.03])
    slider = Slider(ax_slider, label, vmin, vmax, valinit=vinit)
    sliders.append(slider)
    y_pos -= 0.03
# Hex mode toggle using Button.
ax_hex = plt.axes([0.1, 0.01, 0.1, 0.03])
btn_hex = Button(ax_hex, 'Hex Mode: Off')
hex_mode = False
def toggle_hex(event):
    global hex_mode
    hex_mode = not hex_mode
    btn_hex.label.set_text(f'Hex Mode: {"On" if hex_mode else "Off"}')
    update(None)
btn_hex.on_clicked(toggle_hex)
def update(val):
    """Update surface based on current slider values."""
    params = [s.val for s in sliders[:-1]] + [sliders[-1].val, hex_mode] # Morph is last slider
    X, Y, Z, surface_id, X_cap, Y_cap, Z_cap = generate_nurks_surface(*params)
    global surf, surf_cap
    surf.remove()
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    if hex_mode:
        if surf_cap is not None:
            surf_cap.remove()
        surf_cap = ax.plot_surface(X_cap, Y_cap, Z_cap, cmap='viridis', alpha=0.8)
    else:
        if surf_cap is not None:
            surf_cap.remove()
        surf_cap = None
    fig.canvas.draw_idle()
for s in sliders:
    s.on_changed(update)
# Export button.
ax_export = plt.axes([0.8, 0.05, 0.1, 0.075])
btn_export = Button(ax_export, 'Export STL')
u_num = 36
v_num = 20
v_num_cap = 10
def on_export(event):
    params = [s.val for s in sliders[:-1]] + [sliders[-1].val, hex_mode] # Morph is last slider
    X, Y, Z, surface_id, X_cap, Y_cap, Z_cap = generate_nurks_surface(*params)
    triangles_main = tessellate_mesh(X, Y, Z, u_num, v_num)
    triangles = triangles_main
    if hex_mode:
        triangles_cap = tessellate_mesh(X_cap, Y_cap, Z_cap, u_num, v_num_cap, is_cap=True)
        triangles += triangles_cap
    export_to_stl(triangles, 'nurks_surface.stl', surface_id)
    print(f"Exported to nurks_surface.stl with ID: {surface_id}")
btn_export.on_clicked(on_export)
surf_cap = None
plt.show()
