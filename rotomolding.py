# rotomolding.py
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
from rig import Rig

def simulate_rotomolding(mass=1.0, mold_radius=0.5, rpm=1.0, time_step=0.1, steps=60):
    """
    Simulates rotational molding with centrifugal force distribution.
    - mass: Material mass in kg (default 1.0).
    - mold_radius: Mold radius in meters (default 0.5).
    - rpm: Angular velocity in RPM (default 1.0).
    - time_step: Simulation time step in seconds (default 0.1).
    - steps: Number of simulation steps (default 60).
    Returns: List of centrifugal forces in Newtons.
    """
    omega = rpm * 2 * np.pi / 60  # Convert RPM to rad/s
    rig = Rig()
    forces = []
    for step in range(steps):
        # Ramp-up angular velocity (simulates startup)
        current_omega = omega * min(1.0, step / (steps / 2))
        force = mass * mold_radius * current_omega**2
        forces.append(force)
        rig.log("Rotomolding simulation", step=step, force=force, omega=current_omega)
    
    return forces

# Example usage
if __name__ == "__main__":
    forces = simulate_rotomolding()
    print(f"Rotomolding centrifugal forces: {forces[:5]}...")
