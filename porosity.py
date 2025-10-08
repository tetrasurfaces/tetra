# porosity.py
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

import hashlib
import numpy as np

def porosity_hashing(grid, void_threshold=0.3):
    """
    Discretizes pores into hashed grids for porosity simulation.
    - grid: 3D numpy array representing the tetrahedral mesh.
    - void_threshold: Porosity threshold for hashing voids (default 0.3 for 30% void growth).
    Returns: Dict of hashed voids with keys as hash(node) and values as void volumes.
    """
    hashed_voids = {}
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                if grid[i, j, k] > void_threshold:
                    node_str = f"{i}_{j}_{k}"
                    node_hash = hashlib.sha256(node_str.encode()).hexdigest()
                    void_volume = grid[i, j, k] * (1 - void_threshold)  # Simplified volume calculation
                    hashed_voids[node_hash] = void_volume
    return hashed_voids

# Example usage
if __name__ == "__main__":
    # Mock grid for testing (3D array simulating porosity)
    mock_grid = np.random.rand(10, 10, 10)  # Random porosity values between 0 and 1
    voids = porosity_hashing(mock_grid)
    print(f"Hashed voids: {len(voids)}")
