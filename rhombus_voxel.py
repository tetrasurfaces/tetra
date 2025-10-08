# rhombus_voxel.py
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

def generate_rhombus_voxel(grid_size=10, rhombus_angle=60):
    """
    Generates a 3D rhombohedral voxel grid for material simulations.
    - grid_size: Number of voxels per dimension (default 10).
    - rhombus_angle: Angle between rhombohedron edges in degrees (default 60).
    Returns: 3D numpy array representing the voxel grid and hashed voids.
    """
    # Initialize cubic grid, then transform to rhombohedral
    grid = np.random.rand(grid_size, grid_size, grid_size)  # Random material density
    # Apply rhombohedral transformation (simplified shear)
    shear_matrix = np.array([
        [1, np.cos(np.radians(rhombus_angle)), 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    transformed_grid = np.tensordot(grid, shear_matrix, axes=0).mean(axis=-1)
    
    # Hash voids for porosity
    hashed_voids = porosity_hashing(transformed_grid, void_threshold=0.3)
    
    return transformed_grid, hashed_voids

# Example usage
if __name__ == "__main__":
    voxel_grid, voids = generate_rhombus_voxel()
    print(f"Rhombus voxel grid shape: {voxel_grid.shape}")
    print(f"Number of hashed voids: {len(voids)}")
