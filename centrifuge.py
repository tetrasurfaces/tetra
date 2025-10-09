# centrifuge.py
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
from particle_vector import track_particle_vector

def simulate_centrifuge_emulsification(droplet_radius=1e-6, density_diff=100, viscosity=0.001, rpm=1000, radius=0.1, time_step=0.1, steps=300):
    """
    Simulates centrifugal emulsification for paint mixing using modified Stokes' law.
    - droplet_radius: Droplet radius in meters (default 1e-6 m).
    - density_diff: Density difference between phases in kg/m³ (default 100).
    - viscosity: Fluid viscosity in Pa·s (default 0.001).
    - rpm: Angular velocity in RPM (default 1000).
    - radius: Centrifuge radius in meters (default 0.1).
    - time_step: Simulation time step in seconds (default 0.1).
    - steps: Number of simulation steps (default 300).
    Returns: List of separation distances in meters.
    """
    omega = rpm * 2 * np.pi / 60  # Convert RPM to rad/s
    rig = Rig()
    distances = []
    for step in range(steps):
        # Modified Stokes' law: v = (2/9) * (rho_p - rho_f) * r^2 * omega^2 * R / eta
        velocity = (2/9) * density_diff * droplet_radius**2 * omega**2 * radius / viscosity
        distance = velocity * time_step
        distances.append(distance)
        # Log to rig
        rig.log("Centrifuge emulsification", step=step, velocity=velocity, distance=distance)
        # Track particle vector for emulsion stability
        track_particle_vector([(step * time_step, 0, distance, "emulsion")])
    
    return distances

# Example usage
if __name__ == "__main__":
    distances = simulate_centrifuge_emulsification()
    print(f"Centrifuge separation distances: {distances[:5]}...")
