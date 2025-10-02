# gimbal.py
# Copyright 2025 Todd Hutchinson, Beau Ayres, Anonymous
# Beau Ayres owns the IP of Sierpinski mesh as surface detail
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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Voronoi, Delaunay  # For Voronoi hex integration
from tetras import fractal_tetra
from nurks_surface import generate_nurks_surface, u_num, v_num
from tessellations import tessellate_hex_mesh, build_mail
# Global constants
V_NUM_CAP = 10

# Export function
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
                normal = np.array([0.0, 0.0, 1.0])  # Default upward normal.
            f.write(struct.pack('<3f', *normal))
            for p in tri:
                f.write(struct.pack('<3f', *p[1:]))
            f.write(struct.pack('<H', 0))  # Attribute byte count.

# Initial parameters
init_ns_diam = 1.0
init_sw_ne_diam = 1.0
init_nw_se_diam = 1.0
init_twist = 0.0
init_amplitude = 0.3
init_radii = 1.0
init_kappa = 1.0
init_height = 1.0
init_inflection = 0.5
init_morph = 0.0
init_hex_mode = False

# Generate initial surface
X, Y, Z, surface_id, X_cap, Y_cap, Z_cap = generate_nurks_surface(
    ns_diam=init_ns_diam, sw_ne_diam=init_sw_ne_diam, nw_se_diam=init_nw_se_diam,
    twist=init_twist, amplitude=init_amplitude, radii=init_radii, kappa=init_kappa,
    height=init_height, inflection=init_inflection, morph=init_morph, hex_mode=init_hex_mode
)

# Create the figure and 3D axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'NURKS Surface (ID: {surface_id})')

# Global artists for dynamic updates
surf = None
wire = None
surf_cap = None
wire_cap = None

# Initial plot with wireframe to visualize kappa_grid spacing
surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
wire = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='black', linewidth=0.5, alpha=0.5)

# Set axis limits based on initial data
ax.set_xlim(np.min(X), np.max(X))
ax.set_ylim(np.min(Y), np.max(Y))
ax.set_zlim(np.min(Z), np.max(Z))

# Adjust layout for sliders and buttons
plt.subplots_adjust(left=0.25, bottom=0.25, right=0.95, top=0.95)

# Horizontal sliders (bottom)
ax_ns_diam = plt.axes([0.25, 0.20, 0.65, 0.03])
s_ns_diam = Slider(ax_ns_diam, 'NS Diam', 0.1, 2.0, valinit=init_ns_diam)

ax_sw_ne_diam = plt.axes([0.25, 0.17, 0.65, 0.03])
s_sw_ne_diam = Slider(ax_sw_ne_diam, 'SW-NE Diam', 0.1, 2.0, valinit=init_sw_ne_diam)

ax_nw_se_diam = plt.axes([0.25, 0.14, 0.65, 0.03])
s_nw_se_diam = Slider(ax_nw_se_diam, 'NW-SE Diam', 0.1, 2.0, valinit=init_nw_se_diam)

ax_twist = plt.axes([0.25, 0.11, 0.65, 0.03])
s_twist = Slider(ax_twist, 'Twist', -np.pi, np.pi, valinit=init_twist)

ax_amplitude = plt.axes([0.25, 0.08, 0.65, 0.03])
s_amplitude = Slider(ax_amplitude, 'Amplitude', 0.0, 1.0, valinit=init_amplitude)

ax_radii = plt.axes([0.25, 0.05, 0.65, 0.03])
s_radii = Slider(ax_radii, 'Radii', 0.5, 2.0, valinit=init_radii)

ax_kappa = plt.axes([0.25, 0.02, 0.65, 0.03])
s_kappa = Slider(ax_kappa, 'Kappa', 0.1, 5.0, valinit=init_kappa)

# Vertical sliders (left side)
ax_height = plt.axes([0.01, 0.25, 0.0225, 0.63])
s_height = Slider(ax_height, 'Height', 0.1, 2.0, valinit=init_height, orientation='vertical')

ax_inflection = plt.axes([0.04, 0.25, 0.0225, 0.63])
s_inflection = Slider(ax_inflection, 'Inflection', 0.0, 1.0, valinit=init_inflection, orientation='vertical')

ax_morph = plt.axes([0.07, 0.25, 0.0225, 0.63])
s_morph = Slider(ax_morph, 'Morph', 0.0, 2.0, valinit=init_morph, orientation='vertical')

# Buttons (bottom right)
ax_hex = plt.axes([0.025, 0.025, 0.1, 0.075])
b_hex = Button(ax_hex, 'Hex Mode: Off')

ax_export = plt.axes([0.025, 0.11, 0.1, 0.075])
btn_export = Button(ax_export, 'Export STL')

# Update function (remove and recreate artists to avoid clearing axes)
def update(val):
    global surf, wire, surf_cap, wire_cap
    ns_diam = s_ns_diam.val
    sw_ne_diam = s_sw_ne_diam.val
    nw_se_diam = s_nw_se_diam.val
    twist = s_twist.val
    amplitude = s_amplitude.val
    radii = s_radii.val
    kappa = s_kappa.val
    height = s_height.val
    inflection = s_inflection.val
    morph = s_morph.val
    hex_mode = init_hex_mode

    X, Y, Z, surface_id, X_cap, Y_cap, Z_cap = generate_nurks_surface(
        ns_diam=ns_diam, sw_ne_diam=sw_ne_diam, nw_se_diam=nw_se_diam,
        twist=twist, amplitude=amplitude, radii=radii, kappa=kappa,
        height=height, inflection=inflection, morph=morph, hex_mode=hex_mode
    )

    # Remove old main surface artists
    if surf is not None:
        surf.remove()
    if wire is not None:
        wire.remove()

    # Plot new main surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
    wire = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='black', linewidth=0.5, alpha=0.5)

    # Handle cap if in hex mode
    if hex_mode and X_cap is not None:
        if surf_cap is not None:
            surf_cap.remove()
        if wire_cap is not None:
            wire_cap.remove()
        surf_cap = ax.plot_surface(X_cap, Y_cap, Z_cap, cmap='plasma', linewidth=0, antialiased=False)
        wire_cap = ax.plot_wireframe(X_cap, Y_cap, Z_cap, rstride=1, cstride=1, color='black', linewidth=0.5, alpha=0.5)
    else:
        if surf_cap is not None:
            surf_cap.remove()
            surf_cap = None
        if wire_cap is not None:
            wire_cap.remove()
            wire_cap = None

    ax.set_title(f'NURKS Surface (ID: {surface_id})')
    ax.set_xlim(np.min(X), np.max(X))
    ax.set_ylim(np.min(Y), np.max(Y))
    ax.set_zlim(np.min(Z), np.max(Z))
    fig.canvas.draw_idle()

# Toggle hex_mode function
def toggle_hex(event):
    global init_hex_mode
    init_hex_mode = not init_hex_mode
    b_hex.label.set_text(f'Hex Mode: {"On" if init_hex_mode else "Off"}')
    update(None)

b_hex.on_clicked(toggle_hex)

# Export function
def on_export(event):
    global init_hex_mode
    ns_diam = s_ns_diam.val
    sw_ne_diam = s_sw_ne_diam.val
    nw_se_diam = s_nw_se_diam.val
    twist = s_twist.val
    amplitude = s_amplitude.val
    radii = s_radii.val
    kappa = s_kappa.val
    height = s_height.val
    inflection = s_inflection.val
    morph = s_morph.val
    hex_mode = init_hex_mode

    X, Y, Z, surface_id, X_cap, Y_cap, Z_cap = generate_nurks_surface(
        ns_diam=ns_diam, sw_ne_diam=sw_ne_diam, nw_se_diam=nw_se_diam,
        twist=twist, amplitude=amplitude, radii=radii, kappa=kappa,
        height=height, inflection=inflection, morph=morph, hex_mode=hex_mode
    )

    triangles_main = tessellate_hex_mesh(X, Y, Z, U_NUM, V_NUM)
    triangles = triangles_main
    if hex_mode and X_cap is not None:
        triangles_cap = tessellate_hex_mesh(X_cap, Y_cap, Z_cap, U_NUM, V_NUM_CAP, is_cap=True)
        triangles += triangles_cap

    filename = 'nurks_surface.stl'
    export_to_stl(triangles, filename, surface_id)
    print(f"Exported to {filename} with ID: {surface_id}")

btn_export.on_clicked(on_export)

# Connect sliders to update function
s_ns_diam.on_changed(update)
s_sw_ne_diam.on_changed(update)
s_nw_se_diam.on_changed(update)
s_twist.on_changed(update)
s_amplitude.on_changed(update)
s_radii.on_changed(update)
s_kappa.on_changed(update)
s_height.on_changed(update)
s_inflection.on_changed(update)
s_morph.on_changed(update)

plt.show()
