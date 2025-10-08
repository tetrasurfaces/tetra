# simulate_welding.py
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
from ribit import mesh  # Assume ribit.py has mesh function
from telemetry import log  # Assume telemetry.py has log function

def simulate_welding(bead_length=9, preheat=False, num_passes=3):
    """
    Simulates welding sequence with bead length, preheating, and logging.
    - bead_length: Length of each bead in inches (default 9).
    - preheat: Boolean to toggle preheating (default False).
    - num_passes: Number of welding passes (default 3).
    Returns: Deformation map (numpy array).
    """
    # Generate beam mesh
    beam_mesh = mesh("I-beam", dimensions=(8, 32 * 12))  # 8-inch web, 32-foot length in inches
    
    # Preheating simulation
    if preheat:
        log("Preheating to 150C")
        beam_mesh['temp'] = np.full(beam_mesh.shape, 150)  # Uniform preheat
    
    # Welding passes
    deformation = np.zeros(num_passes)
    for pass_num in range(num_passes):
        # Simulate bead application
        heat_input = 90000 * (bead_length / 39.37)  # Joules per meter to per bead
        beam_mesh['temp'] += heat_input / beam_mesh.size  # Simplified heat addition
        # Simulate warping
        warping = np.random.rand() * (bead_length / 10)  # Random warping based on bead length
        deformation[pass_num] = warping
        log(f"Pass {pass_num + 1}: Bead length {bead_length} in, Warping {warping:.2f} mm")
    
    # Cooling simulation
    log("Cooling to room temp")
    beam_mesh['temp'] = np.full(beam_mesh.shape, 20)  # Cool to 20C
    
    return deformation

# Example usage
if __name__ == "__main__":
    def_map_preheat = simulate_welding(bead_length=12, preheat=True)
    def_map_no_preheat = simulate_welding(bead_length=12, preheat=False)
    print(f"Preheat deformation: {def_map_preheat}")
    print(f"No preheat deformation: {def_map_no_preheat}")
