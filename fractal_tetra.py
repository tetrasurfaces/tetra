# fractal_tetra.py
# Copyright 2025 Beau Ayres
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
from porosity_hashing import porosity_hashing
from rhombus_voxel import generate_rhombus_voxel

def generate_fractal_tetra(grid_size=50, levels=3, porosity_threshold=0.3, use_rhombus=False):
    """
    Generates fractal tetrahedral patterns with optional rhombohedral voxels.
    - grid_size: Initial grid dimension (default 50).
    - levels: Fractal recursion levels (default 3).
    - porosity_threshold: Threshold for porosity hashing (default 0.3).
    - use_rhombus: Use rhombohedral voxels if True (default False).
    Returns: Fractal grid and hashed porosity.
    """
    if use_rhombus:
        grid, hashed_porosity = generate_rhombus_voxel(grid_size, rhombus_angle=60)
    else:
        grid = np.random.rand(grid_size, grid_size, grid_size)
        # Recursively apply fractal levels
        for level in range(levels):
            grid = np.pad(grid, pad_width=grid_size // (2 ** level), mode='symmetric')
        grid = grid * (1 - porosity_threshold) + np.random.rand(*grid.shape) * porosity_threshold
        hashed_porosity = porosity_hashing(grid, porosity_threshold)
    
    return grid, hashed_porosity

# Example usage
if __name__ == "__main__":
    fractal_grid, hashed_voids = generate_fractal_tetra(use_rhombus=True)
    print(f"Fractal grid shape: {fractal_grid.shape}")
    print(f"Number of hashed voids: {len(hashed_voids)}")
