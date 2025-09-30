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
from matplotlib import cm
from matplotlib.widgets import Slider, Button, TextBox
import hashlib
from decimal import Decimal, getcontext
import warnings
from matplotlib import MatplotlibDeprecationWarning
import struct
import base64

# Import mpld3 if available for HTML export
try:
    import mpld3
    MPLD3_AVAILABLE = True
except ImportError:
    print("mpld3 not installed. HTML export will be skipped. Install mpld3 with 'pip install mpld3' to enable.")
    MPLD3_AVAILABLE = False
from temperature_salt import secure_hash_two
from kappawise import kappa_coord
from nurks_surface import bspline_basis, bspline_basis_periodic, custom_interoperations_green_curve
# Set precision for Decimal
getcontext().prec = 28
# Suppress warnings
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
# A3 landscape dimensions (normalized: width long side, height short side)
WIDTH = 420 / 297 # A3 landscape: 420mm width, 297mm height, normalized height=1.0
HEIGHT = 1.0
PURPLE_LINES = [1/3, 2/3] # Dividers on the width
unit_per_mm = 1.0 / 297 # Normalize to A3 short side
scale_label = f"Scale: 1mm = {unit_per_mm:.5f} units (A3 short side = 297mm)"
# Dreyfuss human factors: Optimal eye distance ~20 inches (508mm)
EYE_DISTANCE = 500 * unit_per_mm # Normalized eye distance to viewport
HORIZON_HEIGHT = HEIGHT * 0.5 # Default horizon line at half height
EYE_LINE = HORIZON_HEIGHT # Eye line coincides with horizon
# Golden spiral parameters
PHI = (1 + np.sqrt(5)) / 2
kappa = 1 / PHI
A_SPIRAL = 0.001 # Scaled down slightly from 0.01 to fit better
B_SPIRAL = np.log(PHI) / (np.pi / 2)
# Define κθπ for the green segment
theta_max = kappa * np.pi**2 / PHI
# Compute the full spiral
theta_full = np.linspace(0, 10 * np.pi, 1000)
r_full = A_SPIRAL * np.exp(B_SPIRAL * theta_full)
x_full = r_full * np.cos(theta_full)
y_full = r_full * np.sin(theta_full)
# Compute the green segment (θ from π to 2π)
theta_green = np.linspace(np.pi, 2 * np.pi, 200)
r_green = A_SPIRAL * np.exp(B_SPIRAL * theta_green)
x_green = r_green * np.cos(theta_green)
y_green = r_green * np.sin(theta_green)
# Compute the chord and shift
x1, y1 = x_green[0], y_green[0]
x2, y2 = x_green[-1], y_green[-1]
chord_length = np.abs(x2 - x1)
# Shift so the segment starts at x=0
x_green_shifted = x_green - x1
x_green_final = x_green_shifted
# Scale to match the target chord length (between purple lines)
target_chord = PURPLE_LINES[1] - PURPLE_LINES[0]
scale_factor = target_chord / chord_length if chord_length != 0 else 1.0
x_green_scaled = x_green_final * scale_factor
y_green_scaled = y_green * scale_factor
# Shift to start at the first purple line
x_green_final = x_green_scaled + PURPLE_LINES[0]
# Compute κ at 2πR for the green segment
r_max = A_SPIRAL * np.exp(B_SPIRAL * theta_max)
two_pi_r = 2 * np.pi * r_max
kappa_at_2piR = two_pi_r / PHI
# Define the 52 Mersenne prime exponents (updated with the latest known as of 2025)
mersenne_exponents = [
    2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281,
    3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243,
    110503, 132049, 216091, 756839, 859433, 1257787, 1398269, 2976221, 3021377,
    6972593, 13466917, 20996011, 24036583, 25964951, 30402457, 32582657,
    37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933,
    136279841 # Latest known Mersenne prime exponent as of 2025 (52nd)
]
# Map exponents to x-positions (0 to width)
min_exponent = 2
max_exponent_at_x1 = 1_100_000_000
exponent_range_per_x = (max_exponent_at_x1 - min_exponent) / 1.0
x_positions = [(exponent - min_exponent) / exponent_range_per_x * WIDTH for exponent in mersenne_exponents] # Scale to full WIDTH
# Create the 52 curves data
curves = []
curve_lines = []
for i, (exponent, x_pos) in enumerate(zip(mersenne_exponents, x_positions)):
    scale = x_pos / chord_length if chord_length != 0 else 1.0
    x_new = x_green * scale
    y_new = y_green * scale
    x_new_shifted = x_new - x_new[0]
    curves.append((x_new_shifted, y_new, f"M{exponent}"))
    curve_lines.append(None)
# A3 divisions (297 parts on short side)
division_step = HEIGHT / 297
division_positions = np.arange(0, HEIGHT + division_step, division_step)
# Scale key for the title
scale_key_positions = division_positions[::30] # Every 30mm
scale_key_exponents = [int(min_exponent + (max_exponent_at_x1 - min_exponent) * (y / HEIGHT)) for y in scale_key_positions]
scale_key_text = "Scale (y=0 to 1): " + ", ".join([f"{y:.2f}: {exp:,}" for y, exp in zip(scale_key_positions, scale_key_exponents)])
# Flags for Mersenne primes
flag_length = 0.5
start_y = -0.1
wedge_angles = np.linspace(90, 360, len(curves))
flag_positions = []
annotation_objects = []
harmonic_frequencies = []
circle_markers = []
min_exp = min(mersenne_exponents)
max_exp = max(mersenne_exponents)
log_min = np.log(min_exp)
log_max = np.log(max_exp)
min_freq_exp = -4.459
max_freq_exp = 5.506
exponent_range = max_freq_exp - min_freq_exp
log_range = log_max - log_min
for i, (x_new, y_new, label) in enumerate(curves):
    x_end = x_new[-1]
    y_end = y_new[-1]
    x_start = x_end
    y_start = start_y
    angle = wedge_angles[i]
    flag_x = x_end
    flag_y = start_y - flag_length * (i % 2 + 1) / 2
    flag_positions.append((flag_x, flag_y))
    annotation_objects.append(None)
    # Harmonic frequency mapping
    exp = mersenne_exponents[i]
    log_exp = np.log(exp)
    normalized_log = (log_exp - log_min) / log_range if log_range != 0 else 0
    freq_exp = min_freq_exp + normalized_log * exponent_range
    frequency = np.exp(freq_exp)
    harmonic_frequencies.append(frequency)
    circle_markers.append(None)
# Multi-window setup
# Window 1: 2D Plotter
fig_2d = plt.figure(figsize=(12, 8), num="2D Plotter")
ax_2d = fig_2d.add_subplot(111)
# Window 2: 3D Polyhedron Model
fig_3d = plt.figure(figsize=(10, 6), num="3D Polyhedron Model")
ax_3d = fig_3d.add_subplot(111, projection='3d')
# Window 3: Controls
fig_controls = plt.figure(figsize=(4, 6), num="Controls")
ax_scale = fig_controls.add_axes([0.2, 0.8, 0.6, 0.03])
scale_slider = Slider(ax_scale, 'Scale', 0.5, 2.0, valinit=1.0)
ax_kappa = fig_controls.add_axes([0.2, 0.7, 0.6, 0.03])
kappa_slider = Slider(ax_kappa, 'Kappa', 0.1, 2.0, valinit=kappa)
ax_decay = fig_controls.add_axes([0.2, 0.6, 0.6, 0.03])
decay_slider = Slider(ax_decay, 'Decay', 0.1, 1.0, valinit=0.5)
ax_meta = fig_controls.add_axes([0.2, 0.5, 0.6, 0.03])
meta_textbox = TextBox(ax_meta, 'Meta Value', initial="1")
ax_grid_density = fig_controls.add_axes([0.2, 0.4, 0.6, 0.03])
grid_density_slider = Slider(ax_grid_density, 'Grid Density', 50, 200, valinit=110, valstep=1)
# Add Render Model button
ax_render = fig_controls.add_axes([0.2, 0.3, 0.6, 0.05])
render_button = Button(ax_render, 'Render Model')
# Plot setup for 2D Plotter
max_dimension = max(WIDTH, HEIGHT + 1.5)
padding = 0.5
ax_2d.set_xlim(-padding, max_dimension + padding)
ax_2d.set_ylim(-padding, max_dimension + padding)
# A3 page
ax_2d.plot([0, WIDTH, WIDTH, 0, 0], [0, 0, HEIGHT, HEIGHT, 0], 'k-', label='A3 Page')
ax_2d.plot([0, 1.0, 1.0, 0, 0], [0, 0, HEIGHT, HEIGHT, 0], 'k--', label='A4 Page 1')
ax_2d.plot([1.0, 2.0, 2.0, 1.0, 1.0], [0, 0, HEIGHT, HEIGHT, 0], 'k--', label='A4 Page 2')
for x in PURPLE_LINES:
    ax_2d.plot([x, x], [0, HEIGHT], 'm-', label='Projection Space Dividers' if x == PURPLE_LINES[0] else "")
ax_2d.plot([0, WIDTH], [0, 0], 'r-')
# Function to generate a kappa curve between two points
def generate_kappa_curve(x1, y1, x2, y2, kappa_val, ds=0.02):
    # Compute chord length and initial/final directions
    chord = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if chord == 0:
        return [x1], [y1]
    # Initial direction (assume tangent at start is along the chord)
    theta0 = np.arctan2(y2 - y1, x2 - x1)
    # Arc length (approximate as chord length for simplicity)
    s_total = chord
    s = np.arange(0, s_total + ds, ds)
    # Constant kappa scaled by chord length
    kappa_s = kappa_val / chord
    # Integrate curvature to get tangent angle
    theta = theta0 + kappa_s * s
    # Integrate to get x, y coordinates
    x = np.zeros_like(s)
    y = np.zeros_like(s)
    x[0], y[0] = x1, y1
    for i in range(1, len(s)):
        x[i] = x[i-1] + ds * np.cos(theta[i-1])
        y[i] = y[i-1] + ds * np.sin(theta[i-1])
    # Scale and shift to match endpoint
    x_end, y_end = x[-1], y[-1]
    scale_x = (x2 - x1) / (x_end - x1) if (x_end - x1) != 0 else 1
    scale_y = (y2 - y1) / (y_end - y1) if (y_end - y1) != 0 else 1
    scale = min(scale_x, scale_y)
    x = x1 + (x - x1) * scale
    y = y1 + (y - y1) * scale
    return x, y
# Single continuous green kappa curve
def compute_green_kappa_curve(T=1.0, kappa_val=kappa):
    nodes = [
        (1/3, 0),
        (1/3 + 1/9, 0.1 * T),
        (1/3 + 2/9, 0.1 * T),
        (2/3, 0)
    ]
    x_all, y_all = [], []
    chord_lengths = []
    x_all.append(nodes[0][0])
    y_all.append(nodes[0][1])
    for i in range(len(nodes) - 1):
        x1, y1 = nodes[i]
        x2, y2 = nodes[i+1]
        chord = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        chord_lengths.append(chord)
        x_seg, y_seg = generate_kappa_curve(x1, y1, x2, y2, kappa_val)
        x_all.extend(x_seg[1:])
        y_all.extend(y_seg[1:])
    return np.array(x_all), np.array(y_all), chord_lengths
# Initial green kappa curve
T = 1.0
x_green, y_green, chord_lengths = compute_green_kappa_curve(T)
green_spiral, = ax_2d.plot(x_green, y_green, 'g-', label='Green Kappa Curve')
green_nodes = [ax_2d.plot([node[0]], [node[1]], 'go', markersize=6)[0] for node in [(1/3, 0), (1/3 + 1/9, 0.1), (1/3 + 2/9, 0.1), (2/3, 0)]]
# Protractor elements
protractor_line, = ax_2d.plot([], [], 'c-', label='Protractor Line')
protractor_arc, = ax_2d.plot([], [], 'c--', label='Protractor Arc')
protractor_text = ax_2d.text(0, 0, '', ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
protractor_spiral_2, = ax_2d.plot([], [], 'm-', label='Protractor Spiral 2')
ghost_curves = [ax_2d.plot([], [], 'b--', alpha=0.5)[0] for _ in range(4)]
# Cursor elements
cursor, = ax_2d.plot([], [], 'rx', markersize=10, label='Cursor')
cursor_text = ax_2d.text(0, 0, '', ha='left', va='bottom', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
# Ruler elements
ruler_line, = ax_2d.plot([], [], 'y-', label='Ruler Line')
ruler_text = ax_2d.text(0, 0, '', ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
# Toggle flags
show_protractor = False
show_ruler = False
show_harmonics = False
show_draw = False
show_dimension = False
show_pro_mode = False
# Drawing elements
kappa_nodes = []
kappa_curves = []
ghost_handles = []
selected_node = None
drag_start = None
# Grid lines (toggleable and programmable)
grid_lines = []
show_grid = True
def update_grid(density):
    global grid_lines
    for line in grid_lines:
        line.remove()
    grid_lines = []
    step_x = WIDTH / density
    step_y = HEIGHT / density
    for i in range(density + 1):
        x = i * step_x
        y = i * step_y
        vline, = ax_2d.plot([x, x], [0, HEIGHT], 'k--', alpha=0.3)
        hline, = ax_2d.plot([0, WIDTH], [y, y], 'k--', alpha=0.3)
        grid_lines.extend([vline, hline])
    fig_2d.canvas.draw()
# Initial grid
update_grid(110)
# Toggle grid with 'g'
def toggle_grid(event):
    global show_grid
    if event.key == 'g':
        show_grid = not show_grid
        for line in grid_lines:
            line.set_visible(show_grid)
        fig_2d.canvas.draw()
# Connect toggle grid
fig_2d.canvas.mpl_connect('key_press_event', toggle_grid)
# Update grid with slider
def on_grid_density_change(val):
    update_grid(int(val))
grid_density_slider.on_changed(on_grid_density_change)
# Move legend outside viewport
handles, labels = ax_2d.get_legend_handles_labels()
fig_2d.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
# Maximize windows
fig_2d.canvas.manager.window.showMaximized()
fig_3d.canvas.manager.window.showMaximized()
fig_controls.canvas.manager.window.showMaximized()
# Add Navigation Toolbar (assuming Tk backend; may need adjustment for other backends)
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
toolbar_2d = NavigationToolbar2Tk(fig_2d.canvas, fig_2d.canvas.manager.window)
toolbar_3d = NavigationToolbar2Tk(fig_3d.canvas, fig_3d.canvas.manager.window)
# Render button callback (assuming update_3d_model exists; replace with actual function if different)
def on_render_button(event):
    update_3d_model() # Call your 3D rendering function here; e.g., display_pod_surface() or equivalent
render_button.on_clicked(on_render_button)
# Pick event for Mersenne primes
def on_pick_mersenne(event):
    if isinstance(event.artist, plt.Line2D):
        thisline = event.artist
        ind = event.ind
        if ind is not None and len(ind) > 0:
            i = ind[0]
            exponent = mersenne_exponents[i]
            print(f"Selected Mersenne prime: 2^{exponent} - 1")
            for j, line in enumerate(curve_lines):
                if line is not None:
                    line.set_linewidth(1.0 if j != i else 3.0)
            fig_2d.canvas.draw()
# Click to deselect
def on_click_deselect(event):
    if event.button == 3:  # Right click
        for line in curve_lines:
            if line is not None:
                line.set_linewidth(1.0)
        fig_2d.canvas.draw()
# Toggle protractor
def toggle_protractor(event):
    global show_protractor
    if event.key == 'p':
        show_protractor = not show_protractor
        protractor_line.set_visible(show_protractor)
        protractor_arc.set_visible(show_protractor)
        protractor_text.set_visible(show_protractor)
        protractor_spiral_2.set_visible(show_protractor)
        for ghost in ghost_curves:
            ghost.set_visible(show_protractor)
        print(f"Protractor {'shown' if show_protractor else 'hidden'}")
        fig_2d.canvas.draw()
# Toggle ruler
def toggle_ruler(event):
    global show_ruler
    if event.key == 'r':
        show_ruler = not show_ruler
        ruler_line.set_visible(show_ruler)
        ruler_text.set_visible(show_ruler)
        print(f"Ruler {'shown' if show_ruler else 'hidden'}")
        fig_2d.canvas.draw()
# Click for ruler start
def on_click_ruler(event):
    global ruler_start
    if show_ruler and event.inaxes == ax_2d and event.button == 1:
        ruler_start = (event.xdata, event.ydata)
# Motion for protractor and cursor
def on_motion_protractor(event):
    if event.inaxes != ax_2d:
        return
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    # Update cursor
    cursor.set_data([x], [y])
    # Update protractor if shown
    if show_protractor:
        anchor_x, anchor_y = 0.0, 0.0
        protractor_line.set_data([anchor_x, x], [anchor_y, y])
        dx = x - anchor_x
        dy = y - anchor_y
        angle = np.arctan2(dy, dx) * 180 / np.pi
        mid_x = (anchor_x + x) / 2
        mid_y = (anchor_y + y) / 2
        radius_arc = np.sqrt(dx**2 + dy**2) / 4
        start_angle = 0
        end_angle = angle
        num_points = 100
        theta_arc = np.linspace(np.deg2rad(start_angle), np.deg2rad(end_angle), num_points)
        x_arc = mid_x + radius_arc * np.cos(theta_arc)
        y_arc = mid_y + radius_arc * np.sin(theta_arc)
        protractor_arc.set_data(x_arc, y_arc)
        offsets = [-10, -5, 5, 10]
        for i, offset in enumerate(offsets):
            angle_offset = angle + offset
            x_ghost, y_ghost = compute_curve_points(np.pi, 2 * np.pi, num_points // 2, 1.0, angle_offset)
            ghost_curves[i].set_data(anchor_x + x_ghost, anchor_y + y_ghost)
        line_vec = np.array([x - anchor_x, y - anchor_y])
        line_len = np.sqrt(dx**2 + dy**2)
        if line_len == 0:
            line_len = 1e-10
        normal_vec = np.array([-(y - anchor_y), x - anchor_x]) / line_len
        x_spiral, y_spiral = compute_curve_points(np.pi, 2 * np.pi, num_points, 1.0)
        x_mirrored = []
        y_mirrored = []
        for xs, ys in zip(x_spiral, y_spiral):
            point = np.array([xs, ys])
            v = point - np.array([anchor_x, anchor_y])
            projection = np.dot(v, normal_vec) * normal_vec
            mirrored_point = point - 2 * projection
            x_mirrored.append(mirrored_point[0])
            y_mirrored.append(mirrored_point[1])
        protractor_spiral_2.set_data(x + x_mirrored, y + y_mirrored)
        protractor_text.set_position((mid_x, mid_y))
        protractor_text.set_text(f'Angle: {angle:.2f}°\nκ at 2πR: {kappa_at_2piR:.4f}')
    # Update ruler if shown
    if show_ruler and 'ruler_start' in globals():
        ruler_line.set_data([ruler_start[0], x], [ruler_start[1], y])
        dist = np.sqrt((x - ruler_start[0])**2 + (y - ruler_start[1])**2)
        mid_x_r = (ruler_start[0] + x) / 2
        mid_y_r = (ruler_start[1] + y) / 2
        ruler_text.set_position((mid_x_r, mid_y_r))
        ruler_text.set_text(f'Dist: {dist:.4f}')
    # Update cursor text
    height_factor = y / HEIGHT if HEIGHT != 0 else 0
    baseline_spiral_1 = compute_curve_points(0, np.pi, 100, height_factor)
    x_base_spiral_1, y_base_spiral_1 = baseline_spiral_1
    baseline_chord = np.sqrt((x_base_spiral_1[-1] - x_base_spiral_1[0])**2 + (y_base_spiral_1[-1] - y_base_spiral_1[0])**2)
    baseline_spiral_2 = compute_curve_points(np.pi, 2 * np.pi, 100, height_factor)
    x_base_spiral_2, y_base_spiral_2 = baseline_spiral_2
    baseline_chord_2 = np.sqrt((x_base_spiral_2[-1] - x_base_spiral_2[0])**2 + (y_base_spiral_2[-1] - y_base_spiral_2[0])**2)
    x_start_green = x_green_final[0]
    y_start_green = y_green_scaled[0]
    chord_to_green = np.sqrt((x - x_start_green)**2 + (y - y_start_green)**2)
    text_str = (f'κ: {scale_factor:.4f}\n'
                f'Height Factor: {height_factor:.4f}\n'
                f'Cursor: ({x:.4f}, {y:.4f})\n'
                f'Chord to Green: {chord_to_green:.4f}\n'
                f'Baseline Chord (x=0): {baseline_chord:.4f}\n'
                f'Baseline Chord (x=1): {baseline_chord_2:.4f}')
    cursor_text.set_text(text_str)
    cursor_text.set_position((x + 0.05, y + 0.05))
    fig_2d.canvas.draw_idle()
# Toggle harmonics
def toggle_harmonics(event):
    global show_harmonics
    if event.key == 'h':
        show_harmonics = not show_harmonics
        for text in harmonic_texts:
            text.set_visible(show_harmonics)
        print(f"Harmonic frequencies {'shown' if show_harmonics else 'hidden'}")
        fig_2d.canvas.draw()
# Save plot
def save_plot(event):
    if event.key == 'w':
        plt.savefig("nu_curve.png", dpi=300, bbox_inches='tight')
        print("Plot saved as nu_curve.png")
        if MPLD3_AVAILABLE:
            mpld3.save_html(fig_2d, "nu_curve.html")
            print("Interactive plot saved as nu_curve.html")
        else:
            print("Skipping HTML export because mpld3 is not installed.")
# Connect events
fig_2d.canvas.mpl_connect('pick_event', on_pick_mersenne)
fig_2d.canvas.mpl_connect('button_press_event', on_click_deselect)
fig_2d.canvas.mpl_connect('key_press_event', toggle_protractor)
fig_2d.canvas.mpl_connect('key_press_event', toggle_ruler)
fig_2d.canvas.mpl_connect('button_press_event', on_click_ruler)
fig_2d.canvas.mpl_connect('motion_notify_event', on_motion_protractor)
fig_2d.canvas.mpl_connect('key_press_event', toggle_harmonics)
fig_2d.canvas.mpl_connect('key_press_event', save_plot)
fig_2d.canvas.mpl_connect('key_press_event', toggle_draw)
fig_2d.canvas.mpl_connect('key_press_event', toggle_protractor)
fig_2d.canvas.mpl_connect('key_press_event', toggle_ruler)
fig_2d.canvas.mpl_connect('key_press_event', toggle_dimension)
fig_2d.canvas.mpl_connect('key_press_event', to_construction)
fig_2d.canvas.mpl_connect('key_press_event', hide_show)
fig_2d.canvas.mpl_connect('key_press_event', reset_canvas)
fig_2d.canvas.mpl_connect('key_press_event', save_stl)
fig_2d.canvas.mpl_connect('button_press_event', on_click_protractor)
fig_2d.canvas.mpl_connect('button_press_event', on_click_ruler)
fig_2d.canvas.mpl_connect('button_press_event', on_click_dimension)
fig_2d.canvas.mpl_connect('button_press_event', on_click_draw)
fig_2d.canvas.mpl_connect('motion_notify_event', on_motion)
fig_2d.canvas.mpl_connect('key_press_event', auto_close)
fig_2d.canvas.mpl_connect('key_press_event', toggle_pro_mode)
fig_2d.canvas.mpl_connect('pick_event', on_pick)
fig_2d.canvas.mpl_connect('button_press_event', on_button_press)
fig_2d.canvas.mpl_connect('button_release_event', on_button_release)
# Plot properties
ax_2d.set_xlim(-0.1, WIDTH + 0.1)
ax_2d.set_ylim(-1.5, HEIGHT + 0.1)
ax_2d.set_xlabel('x (Exponents: 2 to 11B)')
ax_2d.set_ylabel('y')
ax_2d.set_title('Golden Spiral with 52 Mersenne Prime Curves on A3 Page\n' + scale_key_text, fontsize=10, pad=20)
ax_2d.grid(True)
ax_2d.set_aspect('equal')
# Display default pod surface and draw default pod
def display_pod_surface():
    # Implement pod surface generation using NURKS or similar
    # Example placeholder: Generate a simple sphere mesh
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax_3d.plot_wireframe(x, y, z, color="r")
    fig_3d.canvas.draw()
def draw_default_pod(ax):
    # Draw a default pod shape in 2D
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.5 + 0.2 * np.sin(6 * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta) + 0.5
    ax.plot(x, y, 'b-', label='Default Pod')
    fig_2d.canvas.draw()
display_pod_surface()
draw_default_pod(ax_2d)
# Update green kappa curve with sliders
def update_green_kappa_curve(val):
    global T, x_green, y_green, chord_lengths
    kappa_val = kappa_slider.val
    decay = decay_slider.val
    x_green_new, y_green_new, chord_lengths_new = compute_green_kappa_curve(T, kappa_val)
    for i in range(len(chord_lengths)):
        chord_lengths[i] = chord_lengths[i] * (1 - decay) + chord_lengths_new[i] * decay
    green_spiral.set_data(x_green_new, y_green_new)
    x_green, y_green, chord_lengths = x_green_new, y_green_new, chord_lengths_new
    nodes = [(1/3, 0), (1/3 + 1/9, 0.1 * T), (1/3 + 2/9, 0.1 * T), (2/3, 0)]
    for i, node in enumerate(nodes):
        green_nodes[i].set_data([node[0]], [node[1]])
    fig_2d.canvas.draw()
kappa_slider.on_changed(update_green_kappa_curve)
decay_slider.on_changed(update_green_kappa_curve)
# Define compute_curve_points (placeholder; implement as needed)
def compute_curve_points(start_theta, end_theta, num_points, scale, angle_offset=0):
    theta = np.linspace(start_theta, end_theta, num_points) + np.deg2rad(angle_offset)
    r = A_SPIRAL * np.exp(B_SPIRAL * theta) * scale
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y
# Toggle draw mode
def toggle_draw(event):
    global show_draw
    if event.key == 'd':
        show_draw = not show_draw
        print(f"Draw mode {'enabled' if show_draw else 'disabled'}")
# Click for draw
def on_click_draw(event):
    if show_draw and event.inaxes == ax_2d and event.button == 1:
        x, y = event.xdata, event.ydata
        kappa_nodes.append((x, y))
        ax_2d.plot(x, y, 'ro')
        if len(kappa_nodes) > 1:
            x1, y1 = kappa_nodes[-2]
            x2, y2 = kappa_nodes[-1]
            x_curve, y_curve = generate_kappa_curve(x1, y1, x2, y2, kappa_slider.val)
            curve, = ax_2d.plot(x_curve, y_curve, 'g-')
            kappa_curves.append(curve)
        fig_2d.canvas.draw()
# Toggle dimension
def toggle_dimension(event):
    global show_dimension
    if event.key == 'm':
        show_dimension = not show_dimension
        print(f"Dimension mode {'enabled' if show_dimension else 'disabled'}")
# Click for dimension
# On click for dimension
def on_click_dimension(event):
    if dimension_active and event.inaxes == ax_2d and event.button == 1:
        if ruler_active and ruler_text:
            dimension_labels.append(ruler_text)
        elif selected_curve:
            x, y = selected_curve.get_data()
            length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            mid_x = np.mean(x)
            mid_y = np.mean(y)
            dim_text = ax_2d.text(mid_x, mid_y + 0.05, f'Len: {length:.4f}', ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
            dimension_labels.append(dim_text)
        fig_2d.canvas.draw()
# Change to construction geometry
def to_construction(event):
    global selected_curve
    if event.key == 'g' and selected_curve:
        selected_curve.set_linestyle('--')
        selected_curve.set_color('gray')
        print("Green curve changed to construction geometry")
        selected_curve = None
        fig_2d.canvas.draw()
# Hide/show
def hide_show(event):
    global hidden_elements, selected_curve
    if event.key == 'h':
        if selected_curve:
            if selected_curve.get_visible():
                selected_curve.set_visible(False)
                hidden_elements.append(selected_curve)
                print("Green curve hidden")
            else:
                selected_curve.set_visible(True)
                if selected_curve in hidden_elements:
                    hidden_elements.remove(selected_curve)
                print("Green curve shown")
            selected_curve = None
        else:
            for elem in hidden_elements:
                elem.set_visible(True)
            hidden_elements.clear()
            print("All hidden elements shown")
        fig_2d.canvas.draw()
# Reset canvas
def reset_canvas(event):
    global drawing_points, kappas, previous_kappa, green_curve_line, vanishing_points, selected_curve, current_vertices, current_faces, last_angle, node_scatter, ghost_handles, is_closed, original_colors
    if event.key == 'e':
        drawing_points = []
        kappas = []
        previous_kappa = 1.0
        if green_curve_line:
            green_curve_line.remove()
            green_curve_line = None
        for node in node_scatter:
            node.remove()
        node_scatter = []
        original_colors = []
        for handle in ghost_handles:
            handle.remove()
        ghost_handles = []
        vanishing_points = []
        selected_curve = None
        ax_3d.cla()
        current_vertices = None
        current_faces = None
        last_angle = 0.0
        is_closed = False
        display_pod_surface()
        print("Canvas reset")
        fig_2d.canvas.draw()
# Save STL on key press
def save_stl(event):
    if event.key == 's':
        export_stl()
# Click protractor
def on_click_protractor(event):
    global protractor_line, protractor_text, last_angle
    if protractor_active and event.inaxes == ax_2d and event.button == 1:
        protractor_points.append((event.xdata, event.ydata))
        if len(protractor_points) == 2:
            x1, y1 = protractor_points[0]
            x2, y2 = protractor_points[1]
            if protractor_line:
                protractor_line.remove()
            protractor_line, = ax_2d.plot([x1, x2], [y1, y2], 'b--')
            dx = x2 - x1
            dy = y2 - y1
            angle = np.arctan2(dy, dx) * 180 / np.pi
            last_angle = angle
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            if protractor_text:
                protractor_text.remove()
            protractor_text = ax_2d.text(mid_x, mid_y, f'Angle: {angle:.2f}°', ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
            protractor_points.clear()
            fig_2d.canvas.draw()
# On motion for dragging and preview
def on_motion(event):
    global previous_kappa, dragging, selected_node_index
    if dragging and pro_mode and selected_node_index != -1 and event.inaxes == ax_2d:
        drawing_points[selected_node_index] = (event.xdata, event.ydata)
        node_scatter[selected_node_index].set_offsets([[event.xdata, event.ydata]])
        recalculate_kappas()
        redraw_green_curve(is_closed=is_closed)
        if is_closed:
            update_3d_model()
        fig_2d.canvas.draw()
        return
    if draw_mode and len(drawing_points) > 0 and event.inaxes == ax_2d and not (protractor_active or ruler_active or dimension_active):
        x, y = event.xdata, event.ydata
        preview_points = drawing_points + [(x, y)]
        preview_kappas = kappas + [curvature]
        close_preview = False
        if len(preview_points) > 2:
            dx_first = x - drawing_points[0][0]
            dy_first = y - drawing_points[0][1]
            dist_first = np.sqrt(dx_first**2 + dy_first**2)
            if dist_first < CLOSE_THRESHOLD:
                preview_points[-1] = drawing_points[0]
                preview_kappas[-1] = curvature
                # Preview kappa1 adjustment for closure
                last_theta = np.sqrt((drawing_points[-1][0] - preview_points[-1][0])**2 + (drawing_points[-1][1] - preview_points[-1][1])**2)
                if last_theta < 1e-10: # Manage '0' snap error in preview
                    last_theta = 1e-10
                decay_factor = np.exp(-last_theta / WIDTH / 20.0)
                preview_kappas[0] = preview_kappas[-1] * decay_factor * curvature
                close_preview = True
        # Compute kappa for preview segment (cursor at theta)
        if len(preview_points) > 1:
            preview_kappa = compute_segment_kappa(preview_points[-2], preview_points[-1], curvature, previous_kappa)
            preview_kappas[-1] = preview_kappa
        x_ghost, y_ghost = custom_interoperations_green_curve(preview_points, preview_kappas, is_closed=close_preview)
        ghost_curve.set_data(x_ghost, y_ghost)
        fig_2d.canvas.draw()
# Auto close on 'c'
def auto_close(event):
    global is_closed, drawing_points, kappas, previous_kappa
    if event.key == 'c' and len(drawing_points) > 2:
        # Adjust kappa1 based on last theta and kappa
        last_theta = np.sqrt((drawing_points[-1][0] - drawing_points[0][0])**2 + (drawing_points[-1][1] - drawing_points[0][1])**2)
        if last_theta < 1e-10: # Manage '0' snap error
            last_theta = 1e-10
        decay_factor = np.exp(-last_theta / WIDTH / 20.0)
        kappas[0] = kappas[-1] * decay_factor * curvature # Affect kappa1 with last kappa and decay
        drawing_points.append(drawing_points[0])
        kappas.append(kappas[0]) # Last kappa inherits first kappa's theta (via same value)
        recalculate_kappas() # Recalculate for closure consistency
        is_closed = True
        redraw_green_curve(is_closed=True) # Use closed NURKS for ellipse conditions
        # Get closed curve
        x_curve, y_curve = green_curve_line.get_data()
        if np.hypot(x_curve[-1] - x_curve[0], y_curve[-1] - y_curve[0]) > 1e-5:
            x_curve = np.append(x_curve, x_curve[0])
            y_curve = np.append(y_curve, y_curve[0])
        ax_3d.cla()
        current_vertices, current_faces = build_mesh(x_curve, y_curve, np.zeros(len(x_curve)))
        verts = [[current_vertices[i] for i in f] for f in current_faces]
        ax_3d.add_collection3d(Poly3DCollection(verts, alpha=0.5, facecolors=cm.viridis(np.linspace(0, 1, len(verts)))))
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title('3D User Model (Compound Curvature with End Caps)')
        fig_3d.canvas.draw()
        print("Polyhedron closed and 3D model generated")
        # Compute and print speeds for user generated curve
        print("User Generated Curve Speeds:")
        speeds = []
        for i in range(len(x_curve)):
            speed = int(hashlib.sha256(f"{x_curve[i]}{y_curve[i]}".encode()).hexdigest()[-4:], 16) % 1000 / 1000.0
            print(f"Point {i}: ({x_curve[i]:.4f}, {y_curve[i]:.4f}), Speed: {speed:.4f}")
            speeds.append(speed)
        # Generate G-Code
        gcode = generate_gcode(x_curve, y_curve, speeds)
        with open('model.gcode', 'w') as f:
            f.write(gcode)
        print("G-Code saved to model.gcode")
        print(gcode)
        fig_2d.canvas.draw()
# Toggle pro mode
def toggle_pro_mode(event):
    global show_pro_mode
    if event.key == 'p':
        show_pro_mode = not show_pro_mode
        print(f"Pro mode {'enabled' if show_pro_mode else 'disabled'}")
def on_pick(event):
    global selected_node_index
    artist = event.artist
    if artist in node_scatter:
        selected_node_index = node_scatter.index(artist)
        artist.set_color('yellow') # Highlight selected node
        if pro_mode:
            show_ghost_handles()
        fig_2d.canvas.draw()
# Show ghost handles (theta points, midpoints between nodes)
def show_ghost_handles():
    global ghost_handles
    for handle in ghost_handles:
        handle.remove()
    ghost_handles = []
    num_points = len(drawing_points) if not is_closed else len(drawing_points) - 1 # Avoid double midpoint on close
    for i in range(num_points):
        next_i = (i + 1) % len(drawing_points) if is_closed else i + 1
        if next_i >= len(drawing_points):
            continue
        mid_x = (drawing_points[i][0] + drawing_points[next_i][0]) / 2
        mid_y = (drawing_points[i][1] + drawing_points[next_i][1]) / 2
        handle = ax_2d.scatter(mid_x, mid_y, color='yellow', s=30, marker='o')
        ghost_handles.append(handle)
    fig_2d.canvas.draw()
# On button press for dragging
def on_button_press(event):
    global dragging
    if pro_mode and selected_node_index != -1 and event.inaxes == ax_2d and event.button == 1:
        dragging = True
# On button release for dragging
def on_button_release(event):
    global dragging
    dragging = False
# Update 3D model (placeholder; implement as needed)
def update_3d_model():
    ax_3d.cla()
    display_pod_surface()
    fig_3d.canvas.draw()
# Run the application
plt.show()
