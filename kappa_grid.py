# kappa_grid.py - Spiral-based grid generation for curvature modulation
# Copyright 2025 Beau Ayres
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
# express written permission from Beau Ayres.
#
# AGPL-3.0-or-later licensed
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np

def kappa_grid(grid_size=100):
    """
    Generate a 3D grid of kappa values for curvature modulation.
    
    Args:
        grid_size (int): Size of the grid for x and y dimensions.
    
    Returns:
        np.ndarray: 3D array of shape (grid_size, grid_size, num_angles) containing kappa values.
    """
    # Golden spiral parameters from tetra.py
    PHI = (1 + np.sqrt(5)) / 2
    kappa = 1 / PHI
    A_SPIRAL = 0.001
    B_SPIRAL = np.log(PHI) / (np.pi / 2)
    
    # Define angle steps for rotation (e.g., 360 degrees in 10 steps)
    num_angles = 10
    angles = np.linspace(0, 2 * np.pi, num_angles)
    
    # Create grid
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Initialize kappa grid
    kappa_grid = np.zeros((grid_size, grid_size, num_angles))
    
    # Compute kappa values based on spiral curvature
    for k in range(num_angles):
        theta = angles[k]
        # Compute radius based on spiral
        r = np.sqrt(X**2 + Y**2)
        r[r == 0] = 1e-10  # Avoid division by zero
        # Scale kappa based on spiral equation
        kappa_values = kappa * A_SPIRAL * np.exp(B_SPIRAL * r)
        kappa_grid[:, :, k] = kappa_values
    
    return kappa_grid
