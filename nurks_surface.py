# Copyright Todd Hutchinson, Beau Ayres, Anonymous
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
import hashlib
import struct
import math
import mpmath
mpmath.mp.dps = 19  # Precision for φ, π.

from kappasha import kappasha256
from math_utils import kappa_calc
from wise_transforms import bitwise_transform, hexwise_transform, hashwise_transform
from green_curve import custom_interoperations_green_curve 
from ribit import ribit_generate
from knots_rops import Knot, Rope, knots_rops_sequence
from left_weighted_scale import left_weighted_scale
from tetras import build_mesh, fractal_tetra  # For Sierpinski tetrahedron (mail mesh)
from regulate_hexagons_on_curve import regulate_hexagons_on_curve
from kappawise import kappa_grid  # For kappa-based grid spacing

u_num = 36
v_num = 20
v_num_cap = 10

def generate_nurks_surface(ns_diam=1.0, sw_ne_diam=1.0, nw_se_diam=1.0, twist=0.0, amplitude=0.3, radii=1.0, kappa=1.0, height=1.0, inflection=0.5, morph=0.0, hex_mode=False):
    """Generate parametric NURKS surface points (X, Y, Z) and copyright hash ID using kappasha256."""
    # 36 nodes for angular control.
    u_num = 36
    v_num = 20
    inner_radius = 0.01  # Small to avoid artefacts.
    u = np.linspace(0, 2 * np.pi, u_num)
    v = kappa_grid(inner_radius, 1, v_num, kappa)  # Use kappa-based grid for v
    U, V = np.meshgrid(u, v)
    if hex_mode:
        # Hexagulation: Stagger alternate rows for hexagonal approximation.
        for i in range(1, v_num, 2):
            U[i, :] += np.pi / u_num / 2  # Stagger by half step.
    # Flower profile with 6 petals.
    petal_amp = amplitude * (1 - V)  # Taper for smaller petals at outer ends (V=1).
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
            num_coarse = 100  # High for circular approximation
        theta_coarse = np.linspace(0, 2 * np.pi, num_coarse, endpoint=False)
        if morph_mode == 1:
            sin_coarse = np.sin(3 * theta_coarse + twist)  # For hex-like (3 petals doubled)
        elif morph_mode == 2:
            sin_coarse = np.zeros(num_coarse)  # No sin for circular
        else:
            sin_coarse = np.sin(6 * theta_coarse + twist)  # Flower
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
    Z = height * (1 - dist ** kappa)  # Inverted V, sharper with higher kappa.
    # Curve radial lines (green curves in diagram) by adding twist modulation.
    curve_factor = 0.1 * amplitude  # Curvature based on amplitude.
    X += curve_factor * np.sin(np.pi * V) * np.cos(U + np.pi/4)  # Curve in SW/NE.
    Y += curve_factor * np.sin(np.pi * V) * np.sin(U + np.pi/4)  # Curve in NW/SE.
    # Hash parameters for copyright ID using kappasha256 (key modulated by kappa).
    param_str = f"{ns_diam},{sw_ne_diam},{nw_se_diam},{twist},{amplitude},{radii},{kappa},{height},{inflection},{morph},{hex_mode}"
    if hex_mode:
        param_str += ',bspline_degree=3,bspline_coarse=36'
    key = hashlib.sha256(struct.pack('f', kappa)).digest() * 2  # 64-byte key from kappa.
    surface_id = kappasha256(param_str.encode('utf-8'), key)[0]  # hash_hex as ID.
    print(f"Surface Copyright ID: {surface_id}")
    # Integrate ribit for center modulation if hex_mode.
    if hex_mode:
        ribit_int, state, color = ribit_generate(param_str)
        print(f"Ribit State: {state}, Color: {color}, Int: {ribit_int}")
        # Use ribit state to modulate cap parameters.
        kappa_cap = 3 + state  # >7th for higher states
        twist_cap = twist + 2 * np.pi * state / 7  # Azimuth change
        mini_factor = 0.1 * (state + 1) / 7  # Scaled mini for ribit variation
        # Generate cap with ribit-modulated params.
        mini_radii = radii * mini_factor
        mini_amplitude = amplitude * mini_factor
        v_num_cap = 10
        v_cap = kappa_grid(0, inner_radius, v_num_cap, kappa_cap)  # Use kappa-based grid for cap v
        U_cap, V_cap = np.meshgrid(u, v_cap)
        if hex_mode:
            for i in range(1, v_num_cap, 2):
                U_cap[i, :] += np.pi / u_num / 2  # Stagger cap too for honeycomb.
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
        sin_variation_main = sin_variation[0, :]  # Angular at boundary
        R_main_inner = radii + petal_amp_main_inner * sin_variation_main
        R_cap = R_cap_base + (R_main_inner[None, :] - R_cap_base) * (V_cap / inner_radius)
        # Deform cap with same scales.
        X_cap = R_cap * V_cap * np.cos(U_cap) * scale_x
        Y_cap = R_cap * V_cap * np.sin(U_cap) * scale_y
        # Curve radial for cap.
        X_cap += curve_factor * np.sin(np.pi * V_cap) * np.cos(U_cap + np.pi/4)
        Y_cap += curve_factor * np.sin(np.pi * V_cap) * np.sin(U_cap + np.pi/4)
        # Z for cap with high power for continuity.
        Z_main_inner = height * (1 - (inner_radius - inflection) ** kappa)  # Approximate, assuming inner small.
        dist_cap = V_cap / inner_radius
        Z_cap = height - (height - Z_main_inner) * dist_cap ** kappa_cap
        # Update param_str for cap
        param_str += f',bspline_degree=3,bspline_coarse=36,ribit_state={state},kappa_cap={kappa_cap},mini_factor={mini_factor}'
        key = hashlib.sha256(struct.pack('f', kappa)).digest() * 2  # 64-byte key from kappa.
        surface_id = kappasha256(param_str.encode('utf-8'), key)[0]  # hash_hex as ID.
        print(f"Surface Copyright ID: {surface_id}")
    else:
        X_cap = None
        Y_cap = None
        Z_cap = None
    return X, Y, Z, surface_id, X_cap, Y_cap, Z_cap
