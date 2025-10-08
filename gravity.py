# gravity.py
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
from rig import Rig  # Import for logging

def simulate_gravity(mass=1.0, g=9.81, time_step=0.1, steps=10):
    """
    Simulates gravity effects on a particle or beam.
    - mass: Mass of the object (default 1.0 kg).
    - g: Gravitational acceleration (default 9.81 m/s²).
    - time_step: Simulation time step in seconds (default 0.1).
    - steps: Number of simulation steps (default 10).
    Returns: List of downward displacements.
    """
    displacements = []
    velocity = 0.0
    rig = Rig()
    for step in range(steps):
        force = mass * g
        acceleration = force / mass
        velocity += acceleration * time_step
        displacement = velocity * time_step
        displacements.append(displacement)
        rig.log("Gravity simulation", step=step, displacement=displacement)
    
    return displacements

# Example usage
if __name__ == "__main__":
    displ = simulate_gravity()
    print(f"Gravity displacements: {displ}")
