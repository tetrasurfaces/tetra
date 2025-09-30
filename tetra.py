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
    update_3d_model()  # Call your 3D rendering function here; e.g., display_pod_surface() or equivalent
render_button.on_clicked(on_render_button)
# ... (The rest of the original code remains unchanged, including event connections, plot properties, and plt.show())
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
display_pod_surface()
draw_default_pod(ax_2d)
plt.show()
