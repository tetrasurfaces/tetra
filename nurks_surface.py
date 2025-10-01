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



def generate_nurks_surface(ns_diam=1.0, sw_ne_diam=1.0, nw_se_diam=1.0, twist=0.0, amplitude=0.3, radii=1.0, kappa=1.0, height=1.0, inflection=0.5, hex_mode=False):
    # Generate parametric surface points and mesh.
    # 36 nodes for angular control.
    u_num = 36
    v_num = 20
    inner_radius = 0.01  # Small to avoid artefacts.

    u = np.linspace(0, 2 * np.pi, u_num)
    v = np.linspace(inner_radius, 1, v_num)

    if hex_mode:
        # Hexagulation: Stagger alternate rows for hexagonal approximation.
        for i in range(1, v_num, 2):
            u[i * u_num:(i+1)*u_num] += np.pi / u_num / 2  # Stagger by half step.

    U, V = np.meshgrid(u, v)

    # Flower profile with 6 petals.
    petal_amp = amplitude * (1 - V)  # Taper for smaller petals at outer ends (V=1).
    R = radii + petal_amp * np.sin(6 * U + twist)

    # Deform with diameters (elliptical/radial influence).
    # NS scales y, SW/NE and NW/SE scale diagonals.
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

    # Hash parameters for copyright ID.
    param_str = f"{ns_diam},{sw_ne_diam},{nw_se_diam},{twist},{amplitude},{radii},{kappa},{height},{inflection},{hex_mode}"
    surface_id = kappa_sha256(param_str, kappa)
    print(f"Surface Copyright ID: {surface_id}")

    return X, Y, Z, surface_id

def tessellate_mesh(X, Y, Z, u_num, v_num):
    # Tessellation: Generate triangles from grid.
    triangles = []
    for i in range(v_num - 1):
        for j in range(u_num):
            p1 = (i * u_num + j, X.flatten()[i * u_num + j], Y.flatten()[i * u_num + j], Z.flatten()[i * u_num + j])
            p2 = (i * u_num + (j + 1) % u_num, X.flatten()[i * u_num + (j + 1) % u_num], Y.flatten()[i * u_num + (j + 1) % u_num], Z.flatten()[i * u_num + (j + 1) % u_num])
            p3 = ((i + 1) * u_num + (j + 1) % u_num, X.flatten()[(i + 1) * u_num + (j + 1) % u_num], Y.flatten()[(i + 1) * u_num + (j + 1) % u_num], Z.flatten()[(i + 1) * u_num + (j + 1) % u_num])
            p4 = ((i + 1) * u_num + j, X.flatten()[(i + 1) * u_num + j], Y.flatten()[(i + 1) * u_num + j], Z.flatten()[(i + 1) * u_num + j])
            # Two triangles per quad.
            triangles.append((p1, p2, p3))
            triangles.append((p1, p3, p4))
    return triangles

def export_to_stl(triangles, filename, surface_id):
    # Export mesh to binary STL with embedded hash in header.
    header = f"ID: {surface_id}".ljust(80, ' ').encode('utf-8')
    num_tri = len(triangles)
    with open(filename, 'wb') as f:
        f.write(header)
        f.write(struct.pack('<I', num_tri))
        for tri in triangles:
            # Compute normal.
            v1 = np.array(tri[1][1:]) - np.array(tri[0][1:])
            v2 = np.array(tri[2][1:]) - np.array(tri[0][1:])
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal /= norm_len
            else:
                normal = np.array([0, 0, 1])
            f.write(struct.pack('<3f', *normal))
            for p in tri:
                f.write(struct.pack('<3f', *p[1:]))
            f.write(struct.pack('<H', 0))  # Attribute byte count.

# Interactive visualization.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z, surface_id = generate_nurks_surface()
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('NURKS Surface')

# Sliders for parameters.
plt.subplots_adjust(bottom=0.35)
ax_ns = plt.axes([0.1, 0.25, 0.65, 0.03])
s_ns = Slider(ax_ns, 'NS Diam', 0.5, 2.0, valinit=1.0)
# Add similar sliders for other params...

def update(val):
    X, Y, Z, _ = generate_nurks_surface(s_ns.val, ... )  # Add all sliders.
    surf.remove()
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    fig.canvas.draw_idle()

s_ns.on_changed(update)
# Add on_changed for others.

# Export button.
ax_export = plt.axes([0.8, 0.05, 0.1, 0.075])
btn_export = Button(ax_export, 'Export STL')
def on_export(event):
    X, Y, Z, surface_id = generate_nurks_surface(...)  # Current params.
    triangles = tessellate_mesh(X, Y, Z, 36, 20)
    export_to_stl(triangles, 'nurks_surface.stl', surface_id)
btn_export.on_clicked(on_export)

plt.show()
