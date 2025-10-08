# tetra_gen.py
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
from porosity import porosity_hashing  # Assume porosity_hashing.py is in the same directory

def generate_fractal_tetra(grid_size=50, levels=3, porosity_threshold=0.3):
    """
    Generates fractal tetrahedral patterns and simulates porosity buildup.
    - grid_size: Initial grid dimension for tetra mesh.
    - levels: Fractal recursion levels.
    - porosity_threshold: Threshold for porosity hashing (default 0.3 for 30% voids).
    Returns: Fractal grid with hashed porosity.
    """
    # Generate base tetra grid (placeholder for fractal generation)
    grid = np.random.rand(grid_size, grid_size, grid_size)  # Random values for porosity simulation
    # Recursively apply fractal levels (simplified)
    for level in range(levels):
        grid = np.pad(grid, pad_width=grid_size // (2 ** level), mode='symmetric')  # Expand grid fractally
    
    # Simulate uneven martensite layers with porosity
    grid = grid * (1 - porosity_threshold) + np.random.rand(*grid.shape) * porosity_threshold
    
    # Hash porosity voids
    hashed_porosity = porosity_hashing(grid, porosity_threshold)
    
    return grid, hashed_porosity

# Example usage
if __name__ == "__main__":
    fractal_grid, hashed_voids = generate_fractal_tetra()
    print(f"Fractal grid shape: {fractal_grid.shape}")
    print(f"Number of hashed voids: {len(hashed_voids)}")
