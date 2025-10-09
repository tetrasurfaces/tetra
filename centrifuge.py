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
from particles import track_particle_vector
from coriolis import simulate_coriolis  # Integrate Coriolis effects

def simulate_centrifuge_emulsification(droplet_radius=1e-6, density_diff=100, viscosity=0.001, rpm=1000, radius=0.1, time_step=0.1, steps=300, latitude=35.0):
    """
    Simulates centrifugal emulsification for paint mixing with Coriolis correction.
    - droplet_radius: Droplet radius in meters (default 1e-6 m).
    - density_diff: Density difference between phases in kg/m³ (default 100).
    - viscosity: Fluid viscosity in Pa·s (default 0.001).
    - rpm: Angular velocity in RPM (default 1000).
    - radius: Centrifuge radius in meters (default 0.1).
    - time_step: Simulation time step in seconds (default 0.1).
    - steps: Number of simulation steps (default 300).
    - latitude: Latitude in degrees for Coriolis effect (default 35.0).
    Returns: List of total displacements (centrifugal + Coriolis) in meters.
    """
    omega = rpm * 2 * np.pi / 60  # Convert RPM to rad/s
    earth_omega = 7.29e-5  # Earth's angular velocity (rad/s)
    coriolis_factor = 2 * earth_omega * np.sin(np.radians(latitude))  # Coriolis parameter
    rig = Rig()
    displacements = []
    velocity = np.array([0.0, 0.0, 0.0])  # Initial velocity (x, y, z)
    
    for step in range(steps):
        # Centrifugal velocity (radial component)
        centrifugal_velocity = (2/9) * density_diff * droplet_radius**2 * omega**2 * radius / viscosity
        # Update radial velocity (simplified 1D along radius)
        velocity[0] += centrifugal_velocity * time_step
        
        # Coriolis correction (affects vertical or lateral deflection)
        coriolis_force = -coriolis_factor * velocity[0]  # Simplified for vertical plane
        coriolis_displacement = coriolis_force * time_step**2 / 2  # Displacement from acceleration
        
        # Total displacement (radial + Coriolis vertical)
        total_displacement = centrifugal_velocity * time_step + coriolis_displacement
        displacements.append(total_displacement)
        
        # Log to rig
        rig.log("Centrifuge emulsification", step=step, centrifugal_velocity=centrifugal_velocity, coriolis_displacement=coriolis_displacement, total_displacement=total_displacement)
        # Track particle vector for emulsion stability
        track_particle_vector([(step * time_step, 0, total_displacement, "emulsion")])
    
    return displacements

# Example usage
if __name__ == "__main__":
    displacements = simulate_centrifuge_emulsification()
    print(f"Centrifuge total displacements: {displacements[:5]}...")
