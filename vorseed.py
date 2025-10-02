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
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import hashlib
import struct
import math
import mpmath
mpmath.mp.dps = 19  # Precision for φ, π.
from kappasha import kappasha256, kappa_calc
from wise_transforms import bitwise_transform, hexwise_transform, hashwise_transform
from id_util_nurks_closure_hex import custom_interoperations_green_curve
from ribit import ribit_generate
from knots_rops import Knot, Rope, knots_rops_sequence
from left_weighted_scale import left_weighted_scale
from tetras import build_mesh, fractal_tetra  # For Sierpinski tetrahedron (mail mesh)
from green_curve import custom_interoperations_green_curve
from scipy.spatial import Voronoi, Delaunay  # For Voronoi hex integration
from regulate_hexagons_on_curve import regulate_hexagons_on_curve
from nurks_surface import generate_nurks_surface

u_num = 36
v_num = 20
v_num_cap = 10

def generate_voronoi_seeds(X, Y, Z, inner_radius, param_str):
    """Modular function to generate Voronoi seeds using regulate_hexagons_on_curve."""
    hex_positions = regulate_hexagons_on_curve(X, Y, Z, inner_radius, param_str)
    seeds = np.array([[pos[0], pos[1]] for pos in hex_positions])  # Extract centers as seeds
    return seeds

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
