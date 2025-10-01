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
import hashlib
import struct
import math
import mpmath
mpmath.mp.dps = 19

from kappasha import kappasha256  # After creating module.

def generate_nurks_surface(ns_diam=1.0, sw_ne_diam=1.0, nw_se_diam=1.0, twist=0.0, amplitude=0.3, radii=1.0, kappa=1.0, height=1.0, inflection=0.5, hex_mode=False):
    """Generate parametric NURKS surface points (X, Y, Z) and copyright hash ID using kappasha256."""
    # 36 nodes for angular control.
    u_num = 36
    v_num = 20
    inner_radius = 0.01  # Small to avoid artefacts.

    u = np.linspace(0, 2 * np.pi, u_num)
    v = np.linspace(inner_radius, 1, v_num)

    U, V = np.meshgrid(u, v)

    if hex_mode:
        # Hexagulation: Stagger alternate rows in U.
        for i in range(1, v_num, 2):
            U[i, :] += np.pi / u_num / 2  # Per row stagger.

    # Flower profile with 6 petals.
    petal_amp = amplitude * (1 - V)  # Taper for smaller petals at outer ends (V=1).
    R = radii + petal_amp * np.sin(6 * U + twist)

    # Deform with diameters (elliptical/radial influence).
    scale_x = (sw_ne_diam + nw_se_diam) / 2
    scale_y = ns_diam
    X = R * V * np.cos(U) * scale_x
    Y = R * V * np.sin(U) * scale_y

    # V-curve: Power-based angulation with inflection.
    dist = np.abs(V - inflection)
    Z = height * (1 - dist ** kappa)  # Inverted V, sharper with higher kappa.

    # Curve radial lines (green curves in diagram) by adding twist modulation.
    curve_factor = 0.1 * amplitude  # Curvature based on amplitude.
    X += curve_factor * np.sin(np.pi * V) * np.cos(U + np.pi/4)  # Curve in SW/NE.
    Y += curve_factor * np.sin(np.pi * V) * np.sin(U + np.pi/4)  # Curve in NW/SE.

    # Hash parameters for copyright ID using kappasha256.
    param_str = f"{ns_diam},{sw_ne_diam},{nw_se_diam},{twist},{amplitude},{radii},{kappa},{height},{inflection},{hex_mode}"
    key = hashlib.sha256(struct.pack('f', kappa)).digest() * 2  # 64-byte key from kappa.
    surface_id = kappasha256(param_str.encode('utf-8'), key)[0]  # hash_hex as ID.
    print(f"Surface Copyright ID: {surface_id}")

    return X, Y, Z, surface_id

def tessellate_mesh(X, Y, Z, u_num, v_num):
    """Tessellation: Generate list of triangles from grid points."""
    triangles = []
    for i in range(v_num - 1):
        for j in range(u_num):
            p1 = (i * u_num + j, X[i, j], Y[i, j], Z[i, j])
            p2 = (i * u_num + (j + 1) % u_num, X[i, (j + 1) % u_num], Y[i, (j + 1) % u_num], Z[i, (j + 1) % u_num])
            p3 = ((i + 1) * u_num + (j + 1) % u_num, X[i + 1, (j + 1) % u_num], Y[i + 1, (j + 1) % u_num], Z[i + 1, (j + 1) % u_num])
            p4 = ((i + 1) * u_num + j, X[i + 1, j], Y[i + 1, j], Z[i + 1, j])
            triangles.append((p1, p2, p3))
            triangles.append((p1, p3, p4))
    return triangles

def export_to_stl(triangles, filename, surface_id):
    """Export mesh to binary STL with embedded hash in header."""
    header = f"ID: {surface_id}".ljust(80, ' ').encode('utf-8')
    num_tri = len(triangles)
    try:
        with open(filename, 'wb') as f:
            f.write(header)
            f.write(struct.pack('<I', num_tri))
            for tri in triangles:
                v1 = np.array(tri[1][1:]) - np.array(tri[0][1:])
                v2 = np.array(tri[2][1:]) - np.array(tri[0][1:])
                normal = np.cross(v1, v2)
                norm_len = np.linalg.norm(normal)
                if norm_len > 0:
                    normal /= norm_len
                else:
                    normal = np.array([0.0, 0.0, 1.0])
                f.write(struct.pack('<3f', *normal))
                for p in tri:
                    f.write(struct.pack('<3f', *p[1:]))
                f.write(struct.pack('<H', 0))
        print(f"Exported {filename}.")
    except Exception as e:
        print(f"Export error: {e}")

# Interactive visualization.
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y, Z, surface_id = generate_nurks_surface()
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_title('Interactive NURKS Surface')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Adjust layout for sliders (increased bottom to fit).
plt.subplots_adjust(left=0.25, bottom=0.4)

# Sliders (adjusted step to avoid overlap).
sliders = []
y_pos = 0.32  # Start higher.
step = 0.035  # Larger step.
for label, vmin, vmax, vinit in slider_params:
    ax_slider = plt.axes([0.1, y_pos, 0.65, 0.03])
    slider = Slider(ax_slider, label, vmin, vmax, valinit=vinit)
    sliders.append(slider)
    y_pos -= step

# Hex mode toggle.
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
    params = [s.val for s in sliders] + [hex_mode]
    X, Y, Z, _ = generate_nurks_surface(*params)
    global surf
    surf.remove()
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    fig.canvas.draw_idle()

for s in sliders:
    s.on_changed(update)

# Export button.
ax_export = plt.axes([0.8, 0.05, 0.1, 0.075])
btn_export = Button(ax_export, 'Export STL')
def on_export(event):
    params = [s.val for s in sliders] + [hex_mode]
    X, Y, Z, surface_id = generate_nurks_surface(*params)
    triangles = tessellate_mesh(X, Y, Z, 36, 20)
    export_to_stl(triangles, 'nurks_surface.stl', surface_id)
btn_export.on_clicked(on_export)

plt.show()
