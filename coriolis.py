# coriolis.py
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
from gravity import simulate_gravity  # Integrate gravity for combined effects

def simulate_coriolis(velocity=np.array([10, 0, 0]), omega=7.29e-5, mass=1.0, steps=10):
    """
    Simulates Coriolis effects on a moving object.
    - velocity: Initial velocity vector (default [10, 0, 0] m/s eastward).
    - omega: Earth's angular velocity (default 7.29e-5 rad/s).
    - mass: Mass of the object (default 1.0 kg).
    - steps: Number of simulation steps (default 10).
    Returns: List of Coriolis force vectors.
    """
    forces = []
    rig = Rig()
    gravity_displ = simulate_gravity(mass=mass, steps=steps)  # Integrate gravity displacements
    for step in range(steps):
        coriolis_force = -2 * mass * np.cross([0, omega, 0], velocity)  # Simplified for northern hemisphere
        forces.append(coriolis_force)
        rig.log("Coriolis simulation", step=step, force=coriolis_force, gravity_displ=gravity_displ[step])
        velocity += coriolis_force / mass  # Update velocity with Coriolis acceleration
    
    return forces

# Example usage
if __name__ == "__main__":
    coriolis_forces = simulate_coriolis()
    print(f"Coriolis forces: {coriolis_forces}")
