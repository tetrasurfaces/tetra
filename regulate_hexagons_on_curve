# Copyright 2025 Todd Hutchinson, Anonymous
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
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
import mpmath
import hashlib
import math
from math_utils import kappa_calc
mpmath.mp.dps = 19
from ribit import ribit_generate
from kappasha import mersenne_fluctuation
PHI_FLOAT = (1 + math.sqrt(5)) / 2  # φ ≈1.618
MODULO = 369  # Cyclic diffusion

def regulate_hexagons_on_curve(X_cap, Y_cap, Z_cap, inner_radius, param_str, speed_threshold=0.5, sub_div=4):
    """Algorithm to regulate hexagon size/position based on curve speed on kappa grid."""
    # Step 1: Ribit gyroscope for azimuth and state
    _, state, _ = ribit_generate(param_str)
    azimuth_base = 2 * np.pi * state / 7  # Gyroscope rotation to reach divisions

    # Step 2: Define kappa grid (7x7 for 7-bit ribit)
    grid_size = 7
    cell_size = inner_radius / grid_size
    grid_x, grid_y = np.meshgrid(np.linspace(-inner_radius/2, inner_radius/2, grid_size), 
                                 np.linspace(-inner_radius/2, inner_radius/2, grid_size))
    
    hex_positions = []  # List of (center_x, center_y, size, azimuth)
    for i in range(grid_size):
        for j in range(grid_size):
            cell_center_x = grid_x[i, j]
            cell_center_y = grid_y[i, j]
            
            # Step 3: Calculate local curve speed (gradient norm at cell)
            # Approximate speed as norm of dZ/dV at nearest point
            closest_idx = np.argmin((X_cap.flatten() - cell_center_x)**2 + (Y_cap.flatten() - cell_center_y)**2)
            v_idx = closest_idx // u_num
            speed = np.gradient(Z_cap[v_idx, :])[0].mean()  # Avg gradient along row as speed proxy
            
            # Step 4: Regulate hex size inversely to speed, weighted by kappa_calc
            kappa_val = kappa_calc(i * j, 0)
            hex_size = cell_size / (1 + abs(speed) * kappa_val / MODULO)  # Smaller for higher speed
            
            # Step 5: Position hex at cell center, rotate by azimuth
            azimuth = azimuth_base + twist  # Twist from params for orientation
            
            # Step 6: Multiple hex per cell if speed > threshold
            if abs(speed) > speed_threshold:
                # Subdivide into sub_div hexagons
                sub_cell_size = hex_size / np.sqrt(sub_div)
                for k in range(sub_div):
                    sub_x = cell_center_x + k * sub_cell_size * np.cos(azimuth + k * 2 * np.pi / sub_div)
                    sub_y = cell_center_y + k * sub_cell_size * np.sin(azimuth + k * 2 * np.pi / sub_div)
                    hex_positions.append((sub_x, sub_y, sub_cell_size, azimuth + k * 2 * np.pi / sub_div))
            else:
                hex_positions.append((cell_center_x, cell_center_y, hex_size, azimuth))
    
    return hex_positions

# Example usage (integrate with NURKS cap)
# Assume X_cap, Y_cap, Z_cap, inner_radius, param_str from generate_nurks_surface
# hex_positions = regulate_hexagons_on_curve(X_cap, Y_cap, Z_cap, inner_radius, param_str)
# For visualization, plot hexagons at positions with size and azimuth (rotation).
