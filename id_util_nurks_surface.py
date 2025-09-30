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
from matplotlib.widgets import Slider
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
# Assuming kappawise.py exists with compute_kappa_grid function; if not, define a placeholder
try:
    from kappawise import compute_kappa_grid
except ImportError:
    def compute_kappa_grid(grid_size):
        # Placeholder: return a dummy 3D array
        return np.random.rand(grid_size, grid_size, 360) # Example shape
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
colors = plt.cm.viridis(np.linspace(0, 1, len(mersenne_exponents)))
for i, (exponent, x_pos) in enumerate(zip(mersenne_exponents, x_positions)):
    scale = x_pos / chord_length if chord_length != 0 else 1.0
    x_new = x_green * scale
    y_new = y_green * scale
    x_new_shifted = x_new - x_new[0]
    curves.append((x_new_shifted, y_new, f"M{exponent}"))
    curve_lines.append(None)
# A4 short edge divisions (110 parts, scaled to first part of WIDTH)
division_step = WIDTH / 2 / 110 # Assume first half is A4-like
division_positions = np.arange(0, WIDTH / 2 + division_step, division_step)
# Scale key for the title
scale_key_positions = division_positions[::10] / (WIDTH / 2) # Normalize to 0-1 for first half
scale_key_exponents = [int(2 + (1_100_000_000 - 2) * x) for x in scale_key_positions]
scale_key_text = "Scale (x=0 to WIDTH/2): " + ", ".join([f"{x:.2f}: {exp:,}" for x, exp in zip(scale_key_positions, scale_key_exponents)])
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
    angle = np.deg2rad(wedge_angles[i])
    x_flag = x_start + flag_length * np.cos(angle)
    y_flag = y_start + flag_length * np.sin(angle)
    exponent = mersenne_exponents[i]
    scaled_exponent = min_freq_exp + (np.log(exponent) - log_min) / log_range * exponent_range
    freq = 440 * 2**scaled_exponent
    harmonic_frequencies.append(freq)
    flag_positions.append((x_end, y_end, x_start, y_start, x_flag, y_flag, label, freq))
    angle_deg = wedge_angles[i]
    if (angle_deg - 90) % 5 == 0:
        angle_rad = np.deg2rad(angle_deg)
        x_marker = x_start + (flag_length * 0.5) * np.cos(angle_rad)
        y_marker = y_start + (flag_length * 0.5) * np.sin(angle_rad)
        circle_markers.append((x_marker, y_marker))
# Global variables for interactive modes
protractor_active = False
ruler_active = False
draw_mode = False
dimension_active = False
pro_mode = False
is_closed = False
selected_curve = None
hidden_elements = []
protractor_points = []
protractor_line = None
protractor_text = None
ruler_points = []
ruler_line = None
ruler_text = None
dimension_labels = []
drawing_points = [] # Kappa nodes (first endpoint of each greenchord)
kappas = [] # Kappa values at each node
node_scatter = [] # List of scatter objects for kappa nodes
original_colors = [] # List to store original colors of nodes
selected_node_index = -1
dragging = False
ghost_handles = [] # List for theta ghost handles
green_curve_line = None # Single plot object for the interoperated greencurve
CLOSE_THRESHOLD = 0.05 # Distance to first point to consider closing
SNAP_THRESHOLD = 0.05 # Threshold for snapping to dividers (UX improvement)
vanishing_points = [] # Vanishing points for each triangulation
previous_kappa = 1.0 # Initial kappa for decay
curvature = 1.0 # Initial curvature (kappa)
height = 0.5 # Initial height for 3D model
num_rings = 20 # Number of loft rings
fractal_level = 3 # Fractal level for flowers
radial_chord = 0.5 # Radial chord for flower
tangential_chord = 0.2 # Tangential chord for flower
height_chord = 0.1 # Height chord for flower
current_vertices = None
current_faces = None
last_angle = 0.0 # Last measured angle from protractor
show_harmonics = False
harmonic_texts = []
annotation_objects = []
# Pre-compute kappa grid
kappa_grid = compute_kappa_grid(grid_size=100)
# Fractal Flower Mesh
def fractal_flower(center, scale, level, all_polygons, rotation_angle=0.0):
    """
    Recursively generates flower-shaped polygons for the surface.
    Collects all base-level flower polygons in all_polygons (list of list of [x,y,z]).
    Uses 36 points for better flower resolution with curved petals.
    Applies rotation to the points.
    Args:
        center: [x, y, z] center of the flower.
        scale: Scale factor for the flower size.
        level: Current recursion depth.
        all_polygons: List to collect all base flower polygons.
        rotation_angle: Rotation angle in radians for the flower.
    """
    rot_cos = np.cos(rotation_angle)
    rot_sin = np.sin(rotation_angle)
    num_points = 37 # 36 points for higher resolution
    t = np.linspace(0, 2 * np.pi, num_points)[:-1]
    r = scale * (radial_chord + tangential_chord * np.sin(6 * t)) # 6 petals, use sin for symmetry if needed
    dx = r * np.cos(t)
    dy = r * np.sin(t)
    dz = scale * height_chord * np.cos(6 * t) # Curved z for surface
    # Apply rotation to dx, dy (around z)
    x_rot = center[0] + dx * rot_cos - dy * rot_sin
    y_rot = center[1] + dx * rot_sin + dy * rot_cos
    z_rot = center[2] + dz
    polygon = [[x_rot[j], y_rot[j], z_rot[j]] for j in range(len(t))]
    all_polygons.append(polygon)
    if level == 0:
        return
    # Add smaller flowers at petal tips
    small_scale = scale / PHI # Golden ratio scale
    for i in range(6):
        theta = i * (2 * np.pi / 6)
        tip_r = scale * (radial_chord + tangential_chord) # Max r for tip
        tip_dx = tip_r * np.cos(theta)
        tip_dy = tip_r * np.sin(theta)
        tip_dz = scale * height_chord
        # Rotate tip offset
        tip_x = center[0] + tip_dx * rot_cos - tip_dy * rot_sin
        tip_y = center[1] + tip_dx * rot_sin + tip_dy * rot_cos
        tip_z = center[2] + tip_dz
        tip_center = [tip_x, tip_y, tip_z]
        fractal_flower(tip_center, small_scale, level - 1, all_polygons, rotation_angle + np.pi)
# Triangulate polygon for rendering (fan triangulation)
def triangulate_poly(poly):
    tris = []
    for i in range(1, len(poly) - 1):
        tris.append([poly[0], poly[i], poly[i+1]])
    return tris
# Hash entropy for lower surface
def hash_entropy(p):
    h_str = f"{p[0]:.6f}{p[1]:.6f}{p[2]:.6f}"
    h = int(hashlib.sha256(h_str.encode()).hexdigest(), 16) % 1000 / 1000.0 * 0.05 - 0.025
    return h
# Build mesh using fractal flower (ties to curve by scaling to curve length)
def build_mesh(x_curve, y_curve, z_curve=None, height=0.5, num_rings=20, num_points=None, fractal_level=3):
    """
    Builds two surfaces meeting at the 3D curve with vertical tangent, inheriting each other's curvature in transition.
    Integrates fractal flower for complex surface detail on caps, scaled by curve length.
    Uses flower modulation in loft rings for interlacing petals.
    Args:
        x_curve, y_curve, z_curve: Curve coordinates.
        height: Height for lofting.
        num_rings: Number of rings for loft.
        num_points: Number of points to sample curve.
        fractal_level: Recursion level for fractal flower.
    Returns:
        vertices (np.array): Array of [x, y, z].
        faces (list): List of [idx1, idx2, idx3].
    """
    if num_points is not None:
        indices = np.linspace(0, len(x_curve) - 1, num_points, dtype=int)
        x_curve = x_curve[indices]
        y_curve = y_curve[indices]
        if z_curve is not None:
            z_curve = z_curve[indices]
    n = len(x_curve)
    if z_curve is None:
        z_curve = np.zeros(n) # Default to flat if no z provided
    center_x = drawing_points[0][0] if drawing_points else np.mean(x_curve) # Datum at kappa node 1 if available
    center_y = drawing_points[0][1] if drawing_points else np.mean(y_curve)
    vertices = []
    faces = []
    # Parting line on 3D curve
    parting_base = len(vertices)
    for i in range(n):
        vertices.append([x_curve[i], y_curve[i], z_curve[i]])
    # Upper surface: rings inward with vertical tangent at edge and flower modulation
    upper_bases = [parting_base]
    for l in range(1, num_rings):
        s = l / (num_rings - 1.0)
        scale = 1 - s**2 # Vertical tangent at s=0 (dr/ds=0)
        g_val = (height / 2) * s**2 # Quadratic for constant curvature
        base = len(vertices)
        upper_bases.append(base)
        for i in range(n):
            vec_x = x_curve[i] - center_x
            vec_y = y_curve[i] - center_y
            norm = np.sqrt(vec_x**2 + vec_y**2)
            if norm > 0:
                dir_x = vec_x / norm
                dir_y = vec_y / norm
            else:
                dir_x = 1.0
                dir_y = 0.0
            theta = np.arctan2(vec_y, vec_x)
            phase = 0.0 # Upper phase
            flower_mod = tangential_chord * np.cos(6 * theta + phase) * s # Modulation increases inward
            r = norm * scale * (radial_chord + flower_mod)
            x = center_x + r * dir_x
            y = center_y + r * dir_y
            z = z_curve[i] * (1 - s) + g_val + height_chord * np.sin(6 * theta + phase)
            vertices.append([x, y, z])
    center_upper_idx = len(vertices)
    vertices.append([center_x, center_y, height / 2])
    # Lower surface: mirrored with phase offset for interlacing and entropy
    lower_bases = [parting_base] # Shared edge
    for l in range(1, num_rings):
        s = l / (num_rings - 1.0)
        scale = 1 - s**2
        g_val = (height / 2) * s**2 # Quadratic for constant curvature (sign same for inheritance magnitude)
        base = len(vertices)
        lower_bases.append(base)
        for i in range(n):
            vec_x = x_curve[i] - center_x
            vec_y = y_curve[i] - center_y
            norm = np.sqrt(vec_x**2 + vec_y**2)
            if norm > 0:
                dir_x = vec_x / norm
                dir_y = vec_y / norm
            else:
                dir_x = 1.0
                dir_y = 0.0
            theta = np.arctan2(vec_y, vec_x)
            phase = np.pi / 6 # Lower phase offset for interlacing
            flower_mod = tangential_chord * np.cos(6 * theta + phase) * s
            r = norm * scale * (radial_chord + flower_mod)
            x = center_x + r * dir_x
            y = center_y + r * dir_y
            z = z_curve[i] * (1 - s) - g_val + hash_entropy([x, y, z]) + height_chord * np.sin(6 * theta + phase)
            vertices.append([x, y, z])
    center_lower_idx = len(vertices)
    vertices.append([center_x, center_y, -height / 2])
    # Faces for upper surface
    for ll in range(len(upper_bases) - 1):
        base = upper_bases[ll]
        next_base = upper_bases[ll + 1]
        for i in range(n):
            next_i = (i + 1) % n
            faces.append([base + i, base + next_i, next_base + next_i])
            faces.append([base + i, next_base + next_i, next_base + i])
    # Faces for lower surface
    for ll in range(len(lower_bases) - 1):
        base = lower_bases[ll]
        next_base = lower_bases[ll + 1]
        for i in range(n):
            next_i = (i + 1) % n
            faces.append([base + i, next_base + i, next_base + next_i])
            faces.append([base + i, next_base + next_i, base + next_i])
    # Integrate fractal flower for caps (no fan, use flower fractals)
    # Compute curve length for scale
    curve_length = np.sum(np.sqrt(np.diff(x_curve)**2 + np.diff(y_curve)**2))
    flower_scale = curve_length * 0.1 if curve_length > 0 else 0.5 # Scale to curve
    all_polygons = [] # List of list of [x,y,z] for each polygon
    # Upper cap flower
    fractal_flower(vertices[center_upper_idx], flower_scale, fractal_level, all_polygons, rotation_angle=np.pi)
    # Lower cap flower
    fractal_flower(vertices[center_lower_idx], flower_scale, fractal_level, all_polygons, rotation_angle=np.pi)
    # Add polygons to mesh (triangulate for rendering)
    for poly in all_polygons:
        base_idx = len(vertices)
        vertices.extend(poly)
        for tri in triangulate_poly(range(len(poly))):
            faces.append([base_idx + tri[0], base_idx + tri[1], base_idx + tri[2]])
    # Convert to numpy array
    vertices = np.array(vertices)
    # Snap to integers if hash ends with 0
    for i in range(len(vertices)):
        v = vertices[i]
        h_str = f"{v[0]:.6f}{v[1]:.6f}{v[2]:.4f}"
        h = hashlib.sha256(h_str.encode()).hexdigest()[-1]
        if h == '0':
            vertices[i] = np.round(vertices[i])
    # Add compound curvature modulation with angle and 3D kappa grid for smooth orthographic projections
    grid_size, _, num_angles = kappa_grid.shape
    angle_idx = int((last_angle / 360) * num_angles) % num_angles
    kappa_slice = kappa_grid[:, :, angle_idx]
    # Normalize vertices to -1 to 1 for grid mapping (assuming curve bounds approx [0, WIDTH] x [0, HEIGHT])
    max_dim = max(np.max(np.abs(vertices[:, 0])), np.max(np.abs(vertices[:, 1])))
    norm_x = np.clip(((vertices[:, 0] / max_dim) + 1) / 2 * (grid_size - 1), 0, grid_size - 1).astype(int)
    norm_y = np.clip(((vertices[:, 1] / max_dim) + 1) / 2 * (grid_size - 1), 0, grid_size - 1).astype(int)
    kappa_mod = kappa_slice[norm_y, norm_x]
    vertices[:, 2] += kappa_mod * 0.1 # Scale z modulation
    vertices[:, 0] += kappa_mod * 0.05 * np.sin(2 * np.pi * vertices[:, 2] / height) # Compound in x
    vertices[:, 1] += kappa_mod * 0.05 * np.cos(2 * np.pi * vertices[:, 2] / height) # Compound in y
    # Make flowers sacrificial: remove flower faces after modulation (assume last added are flowers)
    flower_face_start = len(faces) - len(all_polygons) * 35 # Adjusted for 36 points per flower (approx 35 triangles)
    faces = faces[:flower_face_start]
    return vertices, faces
# NURBS basis function
def nurbs_basis(u, i, p, knots):
    if p == 0:
        return 1.0 if knots[i] <= u <= knots[i+1] else 0.0 # Include = for end
    if knots[i+p] == knots[i]:
        c1 = 0.0
    else:
        c1 = (u - knots[i]) / (knots[i+p] - knots[i]) * nurbs_basis(u, i, p-1, knots)
    if knots[i+p+1] == knots[i+1]:
        c2 = 0.0
    else:
        c2 = (knots[i+p+1] - u) / (knots[i+p+1] - knots[i+1]) * nurbs_basis(u, i+1, p-1, knots)
    return c1 + c2
# Compute NURBS curve point
def nurbs_curve_point(u, control_points, weights, p, knots):
    n = len(control_points) - 1
    x = 0.0
    y = 0.0
    denom = 0.0
    for i in range(n + 1):
        b = nurbs_basis(u, i, p, knots)
        denom += b * weights[i]
        x += b * weights[i] * control_points[i][0]
        y += b * weights[i] * control_points[i][1]
    if denom == 0:
        return 0, 0
    return x / denom, y / denom
# Generate NURBS curve
def generate_nurbs_curve(points, weights, p, knots, num_points=1000):
    u_min, u_max = knots[p], knots[-p-1]
    u_values = np.linspace(u_min, u_max, num_points, endpoint=False)
    curve = [nurbs_curve_point(u, points, weights, p, knots) for u in u_values]
    curve.append(curve[0]) # Append first point for exact closure
    return np.array([list(pt) for pt in curve]) # Convert to np.array of shape (num_points+1, 2)
# Compute golden spiral
def compute_golden_spiral():
    theta = np.linspace(0, 10 * np.pi, 1000)
    r = A_SPIRAL * np.exp(B_SPIRAL * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # Scale down to fit within page bounds
    scale_factor = min(WIDTH, HEIGHT) / (2 * np.max(np.abs([x, y]))) * 0.8 # 80% of max to fit comfortably
    x *= scale_factor
    y *= scale_factor
    return x, y
# Custom interoperations for greencurve using NURBS with local kappa adjustment for closure
def custom_interoperations_green_curve(points, kappas, is_closed=False):
    """
    Custom NURBS curve with endpoint kappa and theta decay, using NURBS for ellipse-like conditions on closure.
    For closed curves, appends points for periodic wrapping to achieve higher continuity at closure.
    """
    if len(points) < 2:
        return np.array([]), np.array([])
    # Dynamically adjust degree for few points to ensure anchoring (line for 2 points)
    degree = min(5, len(points) - 1)
    if is_closed:
        points = points + points[1:degree + 1]
        kappas = kappas + kappas[1:degree + 1]
    x_points = [p[0] for p in points]
    y_points = [p[1] for p in points]
    t = np.cumsum([0] + [np.sqrt((x_points[i+1] - x_points[i])**2 + (y_points[i+1] - y_points[i])**2) for i in range(len(points)-1)])
    t_fine = np.linspace(0, t[-1], 1000) if t[-1] > 0 else np.linspace(0, 1, 1000)
    # Generate knots based on theta (distance), non-uniform for decay, adjusted for higher degree
    knots = [0] * (degree + 1) + list(np.cumsum([kappas[i] for i in range(len(points))])) + [t[-1]] * (degree + 1) # Clamped knots for endpoint interpolation
    x_fine = []
    y_fine = []
    for u in t_fine:
        x_val = 0.0
        y_val = 0.0
        n = len(points) - 1
        for i in range(n + 1):
            b = nurbs_basis(u, i, degree, knots) # Higher degree basis
            weight = kappas[i] if i < len(kappas) else kappas[-1] # Weight by kappa
            x_val += b * x_points[i] * weight
            y_val += b * y_points[i] * weight
        # Theta decay adjustment
        decay = np.exp(-u / t[-1] / 20.0) if t[-1] > 0 else 1.0
        x_val *= decay
        y_val *= decay
        x_fine.append(x_val)
        y_fine.append(y_val)
    x = np.array(x_fine)
    y = np.array(y_fine)
    # Inter-sum operations: after computing y, adjust x based on y (as per request)
    x += 0.05 * y # Example inter-sum: x changed based on y for coupled modulation
    return x, y
# Compute kappa for a segment, second endpoint influences next kappa
def compute_segment_kappa(p1, p2, base_kappa=1.0, prev_kappa=1.0):
    x1, y1 = p1
    x2, y2 = p2
    theta = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) # Theta is distance
    if theta < 1e-10:
        return prev_kappa
    decay_factor = np.exp(-theta / WIDTH / 20.0) # Further reduced decay rate
    return prev_kappa * decay_factor * base_kappa
# Recalculate kappas after changes (e.g., closure or node move) for consistency
def recalculate_kappas():
    global previous_kappa
    previous_kappa = kappas[0] # Start with first kappa
    for i in range(1, len(drawing_points)):
        kappas[i] = compute_segment_kappa(drawing_points[i-1], drawing_points[i], curvature, previous_kappa)
        previous_kappa = kappas[i]
# Update 3D model on changes (e.g., height slider or curve update)
def update_3d_model():
    global current_vertices, current_faces
    if green_curve_line:
        x_curve, y_curve = green_curve_line.get_data()
        if is_closed and np.hypot(x_curve[-1] - x_curve[0], y_curve[-1] - y_curve[0]) > 1e-5:
            x_curve = np.append(x_curve, x_curve[0])
            y_curve = np.append(y_curve, y_curve[0])
        ax_3d.cla()
        current_vertices, current_faces = build_mesh(x_curve, y_curve, np.zeros(len(x_curve)), height=height)
        verts = [[current_vertices[i] for i in f] for f in current_faces]
        ax_3d.add_collection3d(Poly3DCollection(verts, alpha=0.5, facecolors=cm.viridis(np.linspace(0, 1, len(verts)))))
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title('3D User Model (Compound Curvature with End Caps)')
        fig_3d.canvas.draw()
# Golden window calculation
def compute_golden_window(x_spiral, y_spiral):
    idx_crossings = np.where(np.diff(np.sign(x_spiral - PURPLE_LINES[0] * WIDTH)))[0]
    if len(idx_crossings) >= 2:
        y1 = y_spiral[idx_crossings[0]]
        y2 = y_spiral[idx_crossings[1]]
        return np.abs(y2 - y1), min(y1, y2), max(y1, y2)
    return 0, 0, 0
# Compute vanishing point for a triangulation
def compute_vanishing_point(tri_points, eye_distance=EYE_DISTANCE):
    mid_x = np.mean([p[0] for p in tri_points])
    mid_y = np.mean([p[1] for p in tri_points])
    vx = mid_x
    vy = HORIZON_HEIGHT + eye_distance * (mid_y - EYE_LINE) / WIDTH
    return vx, vy
# Redraw green curve
def redraw_green_curve(is_closed=False):
    global green_curve_line
    if green_curve_line:
        green_curve_line.remove()
        green_curve_line = None
    if len(drawing_points) >= 2:
        adjusted_kappas = kappas.copy()
        if is_closed and len(adjusted_kappas) > 1:
            adjusted_kappas[1] = 1.5 * adjusted_kappas[1] # Local kappa adjustment for ellipse conditions
        x_green, y_green = custom_interoperations_green_curve(drawing_points, adjusted_kappas, is_closed=is_closed)
        green_curve_line, = ax_2d.plot(x_green, y_green, 'g-', label='Green Curve' if green_curve_line is None else None)
    fig_2d.canvas.draw()
# Setup figures
fig_2d = plt.figure(figsize=(14, 8))
ax_2d = fig_2d.add_subplot(111)
fig_3d = plt.figure(figsize=(10, 6))
ax_3d = fig_3d.add_subplot(111, projection='3d')
fig_controls = plt.figure(figsize=(4, 8))
ax_curvature = fig_controls.add_axes([0.2, 0.8, 0.6, 0.03])
curvature_slider = Slider(ax_curvature, 'Curvature (kappa)', 0.1, 2.0, valinit=curvature)
ax_height = fig_controls.add_axes([0.2, 0.7, 0.6, 0.03])
height_slider = Slider(ax_height, 'Height', 0.1, 2.0, valinit=height)
ax_rings = fig_controls.add_axes([0.2, 0.6, 0.6, 0.03])
rings_slider = Slider(ax_rings, 'Rings', 5, 50, valinit=num_rings, valstep=1)
ax_level = fig_controls.add_axes([0.2, 0.5, 0.6, 0.03])
level_slider = Slider(ax_level, 'Fractal Level', 0, 5, valinit=fractal_level, valstep=1)
ax_radial = fig_controls.add_axes([0.2, 0.4, 0.6, 0.03])
radial_slider = Slider(ax_radial, 'Radial Chord', 0.1, 1.0, valinit=radial_chord)
ax_tangential = fig_controls.add_axes([0.2, 0.3, 0.6, 0.03])
tangential_slider = Slider(ax_tangential, 'Tangential Chord', 0.0, 0.5, valinit=tangential_chord)
ax_height_chord = fig_controls.add_axes([0.2, 0.2, 0.6, 0.03])
height_chord_slider = Slider(ax_height_chord, 'Height Chord', 0.0, 0.5, valinit=height_chord)
# Plot A3 page
ax_2d.plot([0, WIDTH, WIDTH, 0, 0], [0, 0, HEIGHT, HEIGHT, 0], 'k-', label='A3 Landscape Page')
for x in PURPLE_LINES:
    ax_2d.plot([x * WIDTH, x * WIDTH], [0, HEIGHT], 'm-', label='Purple Dividers' if x == PURPLE_LINES[0] else None)
# Horizon line
horizon_line, = ax_2d.plot([0, WIDTH], [HORIZON_HEIGHT, HORIZON_HEIGHT], 'b:', label='Horizon/Eye Line')
# Golden spiral
x_spiral, y_spiral = compute_golden_spiral()
golden_spiral, = ax_2d.plot(x_spiral + WIDTH/2, y_spiral + HEIGHT/2, 'gold', label='Golden Spiral')
# Golden window
golden_window, y_min, y_max = compute_golden_window(x_spiral + WIDTH/2, y_spiral + HEIGHT/2)
ax_2d.fill_between([PURPLE_LINES[0] * WIDTH - 0.05, PURPLE_LINES[0] * WIDTH + 0.05], y_min, y_max, color='yellow', alpha=0.5, label='Golden Window')
# Ghost curve init
ghost_curve, = ax_2d.plot([], [], 'g--', label='Ghost Curve Preview')
# Control indicators in legend
ax_2d.plot([], [], ' ', label='R: Toggle draw mode')
ax_2d.plot([], [], 'b--', label='A: Toggle protractor')
ax_2d.plot([], [], 'c-', label='M: Toggle measure (ruler)')
ax_2d.plot([], [], ' ', label='D: Toggle dimension')
ax_2d.plot([], [], 'r-', label='C: Close polyhedron (manual)')
ax_2d.plot([], [], ' ', label='Click near first point to close')
ax_2d.plot([], [], ' ', label='Click to select curve')
ax_2d.plot([], [], ' ', label='G: To construction geom')
ax_2d.plot([], [], ' ', label='H: Hide/show')
ax_2d.plot([], [], ' ', label='E: Reset canvas')
ax_2d.plot([], [], ' ', label='S: Export STL')
ax_2d.plot([], [], 'k-', label='Curvature Slider (Controls window)')
# Update curvature
def update_curvature(val):
    global curvature
    curvature = val
    if len(drawing_points) >= 1:
        kappas[-1] = curvature
        recalculate_kappas()
        redraw_green_curve(is_closed=is_closed)
        if is_closed:
            update_3d_model()
    fig_2d.canvas.draw()
curvature_slider.on_changed(update_curvature)
# Update height
def update_height(val):
    global height
    height = val
    if is_closed:
        update_3d_model()
height_slider.on_changed(update_height)
# Update rings
def update_rings(val):
    global num_rings
    num_rings = int(val)
    if is_closed:
        update_3d_model()
rings_slider.on_changed(update_rings)
# Update fractal level
def update_level(val):
    global fractal_level
    fractal_level = int(val)
    if is_closed:
        update_3d_model()
level_slider.on_changed(update_level)
# Update radial chord
def update_radial(val):
    global radial_chord
    radial_chord = val
    if is_closed:
        update_3d_model()
radial_slider.on_changed(update_radial)
# Update tangential chord
def update_tangential(val):
    global tangential_chord
    tangential_chord = val
    if is_closed:
        update_3d_model()
tangential_slider.on_changed(update_tangential)
# Update height chord
def update_height_chord(val):
    global height_chord
    height_chord = val
    if is_closed:
        update_3d_model()
height_chord_slider.on_changed(update_height_chord)
# Toggle draw mode
def toggle_draw(event):
    global draw_mode
    if event.key == 'r':
        draw_mode = not draw_mode
        print(f"Draw mode {'enabled' if draw_mode else 'disabled'}")
        fig_2d.canvas.draw()
# Toggle protractor
def toggle_protractor(event):
    global protractor_active
    if event.key == 'a':
        protractor_active = not protractor_active
        print(f"Protractor tool {'enabled' if protractor_active else 'disabled'}")
        if not protractor_active:
            if protractor_line:
                protractor_line.remove()
                protractor_line = None
            if protractor_text:
                protractor_text.remove()
                protractor_text = None
            protractor_points.clear()
        fig_2d.canvas.draw()
# On click for protractor
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
# Toggle ruler (measure)
def toggle_ruler(event):
    global ruler_active
    if event.key == 'm':
        ruler_active = not ruler_active
        print(f"Measure (ruler) tool {'enabled' if ruler_active else 'disabled'}")
        if not ruler_active:
            if ruler_line:
                ruler_line.remove()
                ruler_line = None
            if ruler_text:
                ruler_text.remove()
                ruler_text = None
            ruler_points.clear()
        fig_2d.canvas.draw()
# On click for ruler
def on_click_ruler(event):
    global ruler_line, ruler_text
    if ruler_active and event.inaxes == ax_2d and event.button == 1:
        ruler_points.append((event.xdata, event.ydata))
        if len(ruler_points) == 2:
            x1, y1 = ruler_points[0]
            x2, y2 = ruler_points[1]
            if ruler_line:
                ruler_line.remove()
            ruler_line, = ax_2d.plot([x1, x2], [y1, y2], 'c-')
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx**2 + dy**2)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            if ruler_text:
                ruler_text.remove()
            ruler_text = ax_2d.text(mid_x, mid_y, f'Dist: {dist:.4f}', ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
            ruler_points.clear()
            fig_2d.canvas.draw()
# Toggle dimension
def toggle_dimension(event):
    global dimension_active
    if event.key == 'd':
        dimension_active = not dimension_active
        print(f"Dimension tool {'enabled' if dimension_active else 'disabled'}")
        fig_2d.canvas.draw()
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
# Toggle pro mode (lock/unlock)
def toggle_pro_mode(event):
    global pro_mode
    if event.key == 'x':
        pro_mode = not pro_mode
        print(f"Pro mode {'locked' if pro_mode else 'unlocked'}")
        fig_2d.canvas.draw()
# On pick event for nodes
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
# Generate G-code for the curve with variable speeds (for 2D plotting/CNC, scaled to mm)
def generate_gcode(x, y, speeds, scale=297):
    """
    Generates simple G-code for linear moves along the curve with variable feedrates.
    Args:
        x (array): X coordinates (normalized).
        y (array): Y coordinates (normalized).
        speeds (list): Normalized speeds (0-1) for each point.
        scale (float): Scale factor to convert normalized units to mm (based on A3 height=297mm).
    Returns:
        str: G-code string.
    """
    gcode = "G21 ; Set units to millimeters\n"
    gcode += "G90 ; Absolute positioning\n"
    # Move to start position (rapid)
    gcode += f"G0 X{x[0] * scale:.2f} Y{y[0] * scale:.2f}\n"
    # Linear moves with varying feedrate
    for i in range(1, len(x)):
        feedrate = speeds[i] * 900 + 100 # Scale speed to 100-1000 mm/min
        gcode += f"G1 X{x[i] * scale:.2f} Y{y[i] * scale:.2f} F{feedrate:.0f}\n"
    return gcode
# Drawing mode: Add kappa nodes and update continuous greencurve
def on_click_draw(event):
    global green_curve_line, selected_curve, previous_kappa, vanishing_points, current_vertices, current_faces, is_closed
    if event.inaxes == ax_2d and event.button == 1:
        x, y = event.xdata, event.ydata
        # Snap to dividers if close (UX improvement for precise alignment)
        for div in PURPLE_LINES:
            div_x = div * WIDTH
            if abs(x - div_x) < SNAP_THRESHOLD:
                x = div_x
                break
        if draw_mode and not (protractor_active or ruler_active or dimension_active):
            # Check if near first point to close
            if len(drawing_points) > 2:
                dx_first = x - drawing_points[0][0]
                dy_first = y - drawing_points[0][1]
                dist_first = np.sqrt(dx_first**2 + dy_first**2)
                if dist_first < CLOSE_THRESHOLD:
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
                    redraw_green_curve(is_closed=True) # Use closed NURBS for ellipse conditions
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
                    return
            # Add new kappa node (first endpoint)
            roygbiv = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
            color = roygbiv[len(drawing_points) % 7]
            node = ax_2d.scatter(x, y, color=color, s=50, picker=True, label='Kappa Node' if len(drawing_points) == 0 else None)
            node_scatter.append(node)
            original_colors.append(color)
            drawing_points.append((x, y))
            kappas.append(curvature)
            if len(drawing_points) > 1:
                previous_kappa = compute_segment_kappa(drawing_points[-2], drawing_points[-1], curvature, previous_kappa)
                redraw_green_curve(is_closed=is_closed)
                if len(drawing_points) >= 2:
                    t = np.linspace(0, 1, 100)
                    x_green, y_green = green_curve_line.get_data()
                    curv = compute_curvature(x_green, y_green, t)
                    print(f"Green curve curvature: Max={curv.max():.4f}, Min={curv.min():.4f}")
            if len(drawing_points) >= 3:
                print("Third point added: Introducing depth and triangulation")
                tri_points = drawing_points[-3:]
                vp = compute_vanishing_point(tri_points)
                vanishing_points.append(vp)
                ax_2d.scatter(vp[0], vp[1], color='purple', s=30, label='Vanishing Point' if len(vanishing_points) == 1 else None)
            fig_2d.canvas.draw()
        elif not draw_mode and not (protractor_active or ruler_active or dimension_active):
            min_dist = float('inf')
            selected_curve = None
            if green_curve_line:
                x_curve, y_curve = green_curve_line.get_data()
                dist = np.min(np.sqrt((x_curve - x)**2 + (y_curve - y)**2))
                if dist < min_dist and dist < 0.05:
                    min_dist = dist
                    selected_curve = green_curve_line
            if selected_curve:
                selected_curve.set_linewidth(3.0)
                print("Green curve selected")
                fig_2d.canvas.draw()
# Close polyhedron (manual trigger)
def close_polyhedron(event):
    if event.key == 'c':
        print("Close via clicking near first point when ghosted")
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
        redraw_green_curve(is_closed=True) # Use closed NURBS for ellipse conditions
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
# Compute curvature for continuity check
def compute_curvature(x, y, t):
    dt = t[1] - t[0]
    dx_dt = np.gradient(x, dt)
    dy_dt = np.gradient(y, dt)
    d2x_dt2 = np.gradient(dx_dt, dt)
    d2y_dt2 = np.gradient(dy_dt, dt)
    numerator = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
    denominator = (dx_dt**2 + dy_dt**2)**1.5
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return numerator / denominator
# Generate base pod curve (closed for boundary surface, now 3D curve)
def generate_pod_curve_closed(num_points=200, phase=0.0):  # Increased num_points for better resolution
    t = np.linspace(0, 2 * np.pi, num_points) # Full closed loop
    r = radial_chord + tangential_chord * np.cos(6 * t + phase) # Flower-like top profile
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = height_chord * np.sin(6 * t + phase) # Add z variation for 3D curve
    return x, y, z
# Function to compute normals
def compute_normal(v1, v2, v3):
    vec1 = v2 - v1
    vec2 = v3 - v1
    normal = np.cross(vec1, vec2)
    norm = np.linalg.norm(normal)
    return normal / norm if norm != 0 else normal
# Export current model to STL
def export_stl():
    global current_vertices, current_faces
    if current_vertices is None or current_faces is None:
        print("No model to export")
        return
    stl_data = b'\x00' * 80 # Header
    stl_data += struct.pack('<I', len(current_faces)) # Number of triangles
    for face in current_faces:
        v1 = current_vertices[face[0]]
        v2 = current_vertices[face[1]]
        v3 = current_vertices[face[2]]
        normal = compute_normal(v1, v2, v3)
        stl_data += struct.pack('<3f', *normal)
        stl_data += struct.pack('<3f', *v1)
        stl_data += struct.pack('<3f', *v2)
        stl_data += struct.pack('<3f', *v3)
        stl_data += b'\x00\x00' # Attribute byte count
    filename = 'model.stl'
    with open(filename, 'wb') as f:
        f.write(stl_data)
    print(f"Saved to {filename}")
    stl_base64 = base64.b64encode(stl_data).decode('utf-8')
    print("Base64 STL:")
    print(stl_base64)
# Save STL on key press
def save_stl(event):
    if event.key == 's':
        export_stl()
# Display pod surface by default in 3D with curvature continuous end caps
def display_pod_surface():
    global current_vertices, current_faces
    x_curve, y_curve, z_curve = generate_pod_curve_closed(200) # Increased for resolution
    current_vertices, current_faces = build_mesh(x_curve, y_curve, z_curve)
    verts = [[current_vertices[i] for i in f] for f in current_faces]
    ax_3d.add_collection3d(Poly3DCollection(verts, alpha=0.5, facecolors=cm.viridis(np.linspace(0, 1, len(verts)))))
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('3D pod Projected Surface (Compound Curvature with End Caps)')
    fig_3d.canvas.draw()
# Draw default pod ellipse as green curve on 2D canvas
def draw_default_pod(ax, color='g'):
    x, y, _ = generate_pod_curve_closed(num_points=36) # Increased to 36 for better flower
    x_control = x[:-1]
    y_control = y[:-1]
    scale = 0.6 # Scale for large curve
    x_control *= scale
    y_control *= scale
    x_control += WIDTH / 2
    y_control += HEIGHT / 2
    points = list(zip(x_control, y_control))
    kappas_pod = [1.0] * len(points)
    x_interp, y_interp = custom_interoperations_green_curve(points, kappas_pod)
    ax.plot(x_interp, y_interp, color=color, linewidth=3, linestyle='-')
    # Compute and print speeds for default curve
    print("Default Curve Speeds:")
    for i in range(len(x_interp)):
        speed = int(hashlib.sha256(f"{x_interp[i]}{y_interp[i]}".encode()).hexdigest()[-4:], 16) % 1000 / 1000.0
        print(f"Point {i}: ({x_interp[i]:.4f}, {y_interp[i]:.4f}), Speed: {speed:.4f}")
# Add Mersenne elements to the plot
# First A4 page (adjusted)
ax_2d.plot([0, WIDTH/2, WIDTH/2, 0, 0], [0, 0, HEIGHT, HEIGHT, 0], 'k--', label='A4 Page 1')
# Second A4 page
ax_2d.plot([WIDTH/2, WIDTH, WIDTH, WIDTH/2, WIDTH/2], [0, 0, HEIGHT, HEIGHT, 0], 'k--', label='A4 Page 2')
# Purple lines (in first A4)
for x in PURPLE_LINES:
    ax_2d.plot([x * (WIDTH/2), x * (WIDTH/2)], [0, HEIGHT], 'm-')
# Red datum line
ax_2d.plot([0, WIDTH], [0, 0], 'r-')
# A4 short edge divisions (first A4 only)
for x in division_positions:
    ax_2d.plot([x, x], [0, 0.02], 'k-', alpha=0.3)
# Plot circle division markers
for x_marker, y_marker in circle_markers:
    ax_2d.plot(x_marker, y_marker, 'k.', markersize=3)
# Full spiral
ax_2d.plot(x_full, y_full, 'k-')
# Green segment
ax_2d.plot(x_green_final, y_green_scaled, 'g-')
# 52 Mersenne prime curves
for i, (x_new, y_new, label) in enumerate(curves):
    line, = ax_2d.plot(x_new, y_new, color=colors[i])
    curve_lines[i] = line
# Flags and staggered labels
label_y_offset = 0.05
for i, (x_end, y_end, x_start, y_start, x_flag, y_flag, label, freq) in enumerate(flag_positions):
    ax_2d.plot([x_end, x_start], [y_end, y_start], 'k--', alpha=0.3)
    ax_2d.plot([x_start, x_flag], [y_start, y_flag], 'k-', alpha=0.5)
    y_label = y_flag - (i % 5) * label_y_offset
    text = ax_2d.text(x_flag, y_label, label, ha='left', va='top', fontsize=6, rotation=45, picker=5)
    harmonic_text = ax_2d.text(x_flag, y_label - 0.1, f"{freq:.1f} Hz", ha='left', va='top', fontsize=6, rotation=45, visible=False)
    annotation_objects.append((text, i))
    harmonic_texts.append(harmonic_text)
# Golden window 1 (vertical at x = 1/3)
idx_crossings_x = np.where(np.diff(np.sign(x_full - PURPLE_LINES[0])))[0]
if len(idx_crossings_x) >= 2:
    y1 = y_full[idx_crossings_x[0]]
    y2 = y_full[idx_crossings_x[1]]
    golden_window_1 = np.abs(y2 - y1)
    print(f"Golden Window 1 at x={PURPLE_LINES[0]}: {golden_window_1:.4f}")
    ax_2d.fill_between([PURPLE_LINES[0] - 0.05, PURPLE_LINES[0] + 0.05], min(y1, y2), max(y1, y2), color='yellow', alpha=0.5)
# Golden window 2 (horizontal at y = 1/3)
idx_crossings_y = np.where(np.diff(np.sign(y_full - 1/3)))[0]
if len(idx_crossings_y) >= 2:
    x1 = x_full[idx_crossings_y[0]]
    x2 = x_full[idx_crossings_y[1]]
    golden_window_2 = np.abs(x2 - x1)
    print(f"Golden Window 2 at y=1/3: {golden_window_2:.4f}")
    ax_2d.fill_betweenx([1/3 - 0.05, 1/3 + 0.05], min(x1, x2), max(x1, x2), color='orange', alpha=0.5)
# Scale label
ax_2d.text(WIDTH, 1.10337, scale_label, ha='right', va='bottom', fontsize=8)
# Update title with scale key
ax_2d.set_title('Golden Spiral with 52 Mersenne Prime Curves on A3 Page\n' + scale_key_text, fontsize=10, pad=20)
# Highlighting functionality for Mersenne labels
highlighted = [None, None]
def on_pick_mersenne(event):
    global highlighted
    artist = event.artist
    for text, idx in annotation_objects:
        if artist == text:
            if highlighted[0] is not None:
                highlighted[0].set_color('black')
                highlighted[0].set_weight('normal')
                curve_lines[highlighted[1]].set_linewidth(1.0)
                curve_lines[highlighted[1]].set_color(colors[highlighted[1]])
            text.set_color('red')
            text.set_weight('bold')
            curve_lines[idx].set_linewidth(2.0)
            curve_lines[idx].set_color('red')
            highlighted = [text, idx]
            fig_2d.canvas.draw()
            break
def on_click_deselect(event):
    global highlighted
    if event.inaxes != ax_2d:
        return
    clicked_on_annotation = False
    for text, idx in annotation_objects:
        if text.contains(event)[0]:
            clicked_on_annotation = True
            break
    if not clicked_on_annotation and highlighted[0] is not None:
        highlighted[0].set_color('black')
        highlighted[0].set_weight('normal')
        curve_lines[highlighted[1]].set_linewidth(1.0)
        curve_lines[highlighted[1]].set_color(colors[highlighted[1]])
        highlighted = [None, None]
        fig_2d.canvas.draw()
# Curve cache for hashing
curve_cache = {}
def compute_curve_points(theta_start, theta_end, num_points, scale_factor, rotation_angle=0):
    # Create a hash key based on parameters
    key = f"{theta_start:.2f}:{theta_end:.2f}:{num_points}:{scale_factor:.4f}:{rotation_angle:.2f}"
    key_hash = hashlib.md5(key.encode()).hexdigest()
    if key_hash in curve_cache:
        return curve_cache[key_hash]
    theta = np.linspace(theta_start, theta_end, num_points)
    r = scale_factor * A_SPIRAL * np.exp(B_SPIRAL * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # Apply rotation
    if rotation_angle != 0:
        angle_rad = np.deg2rad(rotation_angle)
        x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        x, y = x_rot, y_rot
    curve_cache[key_hash] = (x, y)
    return x, y
# Dynamic LOD
def get_num_points_for_curve():
    xlim = ax_2d.get_xlim()
    ylim = ax_2d.get_ylim()
    view_width = xlim[1] - xlim[0]
    view_height = ylim[1] - ylim[0]
    # Base number of points when fully zoomed out
    base_points = 20
    max_points = 200
    # Zoom factor: smaller view range means more zoom
    full_range = WIDTH # Full x-range when zoomed out
    zoom_factor = full_range / view_width
    num_points = int(base_points + (max_points - base_points) * min(zoom_factor / 10, 1))
    return max(base_points, min(max_points, num_points))
# Cursor, spiral, and circumference setup
cursor, = ax_2d.plot([], [], 'ro', markersize=8, label='κ Spiral Cursor', visible=False)
cursor_spiral, = ax_2d.plot([], [], 'g-', alpha=0.5, visible=False)
cursor_circumference = plt.Circle((0, 0), 0, color='b', fill=False, linestyle='--', alpha=0.5, visible=False)
ax_2d.add_patch(cursor_circumference)
cursor_text = ax_2d.text(WIDTH / 2, 1.15, '', ha='center', va='bottom', fontsize=8, visible=False)
baseline_spiral, = ax_2d.plot([], [], 'g-', alpha=0.5, label='Baseline Spiral', visible=False)
baseline_spiral_2, = ax_2d.plot([], [], 'g-', alpha=0.5, label='Baseline Spiral 2', visible=False)
# Crosslines
vertical_line, = ax_2d.plot([], [], 'k--', alpha=0.5, visible=False)
horizontal_line, = ax_2d.plot([], [], 'k--', alpha=0.5, visible=False)
vertical_label = ax_2d.text(target_chord, HEIGHT + 0.05, f'Chord: {target_chord:.4f}', ha='center', va='bottom', fontsize=8, visible=False)
# Protractor elements
protractor_line, = ax_2d.plot([], [], 'b-', alpha=0.8, visible=False)
protractor_text = ax_2d.text(0, 0, '', ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.8), visible=False)
protractor_arc, = ax_2d.plot([], [], 'b-', alpha=0.5, visible=False)
protractor_spiral_2, = ax_2d.plot([], [], 'g-', alpha=0.5, visible=False)
# Baseline angle (grey ghost line)
baseline_angle_line, = ax_2d.plot([0, WIDTH], [0, 0], 'grey', alpha=0.3, linestyle='--', visible=False)
# Swinging ghost curves
ghost_curves = []
for _ in range(4): # ±5°, ±10° (4 curves total)
    line, = ax_2d.plot([], [], 'grey', alpha=0.2, visible=False)
    ghost_curves.append(line)
# Ruler elements
ruler_divisions = []
for _ in range(10): # Up to 10 division markers
    marker, = ax_2d.plot([], [], 'k|', markersize=10, markeredgewidth=2, visible=False)
    ruler_divisions.append(marker)
ruler_vanishing_line, = ax_2d.plot([], [], 'k--', alpha=0.5, visible=False)
# Toggle protractor
def toggle_protractor(event):
    global protractor_active
    if event.key == 'a':
        protractor_active = not protractor_active
        cursor.set_visible(protractor_active)
        cursor_spiral.set_visible(protractor_active)
        cursor_circumference.set_visible(protractor_active)
        cursor_text.set_visible(protractor_active)
        baseline_spiral.set_visible(protractor_active)
        baseline_spiral_2.set_visible(protractor_active)
        vertical_line.set_visible(protractor_active)
        horizontal_line.set_visible(protractor_active)
        vertical_label.set_visible(protractor_active)
        protractor_line.set_visible(protractor_active)
        protractor_text.set_visible(protractor_active)
        protractor_arc.set_visible(protractor_active)
        protractor_spiral_2.set_visible(protractor_active)
        baseline_angle_line.set_visible(protractor_active)
        for curve in ghost_curves:
            curve.set_visible(protractor_active)
        print(f"Protractor tool {'enabled' if protractor_active else 'disabled'}")
        fig_2d.canvas.draw()
# On motion for protractor
def on_motion_protractor(event):
    if not protractor_active or event.inaxes != ax_2d:
        return
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    # Update cursor position
    cursor.set_data([x], [y])
    # Update circumference
    radius = np.sqrt(x**2 + y**2)
    cursor_circumference.set_center((x, y))
    cursor_circumference.set_radius(radius)
    # Dynamic LOD: Adjust number of points based on zoom
    num_points = get_num_points_for_curve()
    # Update cursor spiral
    x_spiral, y_spiral = compute_curve_points(np.pi, 2 * np.pi, num_points, 1.0)
    cursor_spiral.set_data(x + x_spiral, y + y_spiral)
    # Update baseline spiral (indexed at (0,0))
    x_base = 0.0
    scale_factor = (event.xdata / WIDTH) if event.xdata > 0 else 0.01
    scaled_a = A_SPIRAL * scale_factor
    height_factor = (event.ydata / HEIGHT) if event.ydata > 0 else 0.01
    x_base_spiral, y_base_spiral = compute_curve_points(2 * np.pi, np.pi, num_points, scale_factor)
    x_base_spiral = x_base + x_base_spiral * np.abs(np.cos(np.linspace(2 * np.pi, np.pi, num_points)))
    y_base_spiral = y_base_spiral * height_factor
    baseline_spiral.set_data(x_base_spiral, y_base_spiral)
    # Compute the chord length of the baseline spiral
    x_start = x_base_spiral[0]
    y_start = y_base_spiral[0]
    x_end = x_base_spiral[-1]
    y_end = y_base_spiral[-1]
    baseline_chord = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
    # Update second baseline spiral (indexed at (1.0, 0))
    x_base_2 = 1.0
    x_base_spiral_2, y_base_spiral_2 = compute_curve_points(2 * np.pi, np.pi, num_points, scale_factor)
    x_base_spiral_2 = x_base_2 + x_base_spiral_2 * np.abs(np.cos(np.linspace(2 * np.pi, np.pi, num_points)))
    y_base_spiral_2 = y_base_spiral_2 * height_factor
    baseline_spiral_2.set_data(x_base_spiral_2, y_base_spiral_2)
    # Compute the chord length of the second baseline spiral
    x_start_2 = x_base_spiral_2[0]
    y_start_2 = y_base_spiral_2[0]
    x_end_2 = x_base_spiral_2[-1]
    y_end_2 = y_base_spiral_2[-1]
    baseline_chord_2 = np.sqrt((x_end_2 - x_start_2)**2 + (y_end_2 - y_start_2)**2)
    # Update crosslines
    vertical_line.set_data([target_chord, target_chord], [0, HEIGHT])
    vertical_label.set_position((target_chord, HEIGHT + 0.05))
    if y > 0:
        horizontal_line.set_data([0, WIDTH], [y, y])
    else:
        horizontal_line.set_data([], [])
    # Update protractor line (from (0,0) to mouse position)
    anchor_x, anchor_y = 0.0, 0.0
    protractor_line.set_data([anchor_x, x], [anchor_y, y])
    # Compute the angle relative to the baseline (y=0)
    dx = x - anchor_x
    dy = y - anchor_y
    angle = np.arctan2(dy, dx) * 180 / np.pi
    # Update protractor arc
    mid_x = (anchor_x + x) / 2
    mid_y = (anchor_y + y) / 2
    radius_arc = np.sqrt(dx**2 + dy**2) / 4
    start_angle = 0
    end_angle = angle
    theta_arc = np.linspace(np.deg2rad(start_angle), np.deg2rad(end_angle), num_points)
    x_arc = mid_x + radius_arc * np.cos(theta_arc)
    y_arc = mid_y + radius_arc * np.sin(theta_arc)
    protractor_arc.set_data(x_arc, y_arc)
    # Update swinging ghost curves
    offsets = [-10, -5, 5, 10] # Degrees
    for i, offset in enumerate(offsets):
        angle_offset = angle + offset
        x_ghost, y_ghost = compute_curve_points(np.pi, 2 * np.pi, num_points // 2, 1.0, angle_offset)
        ghost_curves[i].set_data(anchor_x + x_ghost, anchor_y + y_ghost)
    # Update protractor spiral at the mouse position
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
    # Update protractor text
    protractor_text.set_position((mid_x, mid_y))
    protractor_text.set_text(f'Angle: {angle:.2f}°\nκ at 2πR: {kappa_at_2piR:.4f}')
    # Calculate chord length from cursor to the start of the green segment
    x_start_green, y_start_green = x_green_final[0], y_green_scaled[0]
    chord_to_green = np.sqrt((x - x_start_green)**2 + (y - y_start_green)**2)
    # Update cursor text
    text_str = (f'κ: {scale_factor:.4f}\n'
                f'Height Factor: {height_factor:.4f}\n'
                f'Cursor: ({x:.4f}, {y:.4f})\n'
                f'Chord to Green: {chord_to_green:.4f}\n'
                f'Baseline Chord (x=0): {baseline_chord:.4f}\n'
                f'Baseline Chord (x=1): {baseline_chord_2:.4f}')
    cursor_text.set_text(text_str)
    fig_2d.canvas.draw()
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
