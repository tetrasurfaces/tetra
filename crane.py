# crane.py
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

def simulate_crane_sway(beam_length, steps, wind_speed=5.0):
    """
    Simulate crane sway displacement based on beam length and wind speed.
    
    Args:
        beam_length (float): Length of the crane beam in meters.
        steps (int): Number of simulation steps.
        wind_speed (float, optional): Wind speed in m/s. Defaults to 5.0.
    
    Returns:
        list: List of displacement values (in meters) for each step.
    """
    # Parameters for harmonic oscillation with damping
    amplitude = 0.1 * beam_length  # Base amplitude proportional to beam length
    frequency = 0.5 / beam_length  # Frequency inversely proportional to length
    damping = 0.1  # Damping factor
    
    displacements = []
    for i in range(steps):
        time = i * 0.1  # Time step of 0.1 seconds
        # Harmonic oscillation with wind-induced amplitude and damping
        displacement = amplitude * np.sin(2 * np.pi * frequency * time) * np.exp(-damping * time)
        # Add wind effect (proportional to wind speed)
        wind_effect = wind_speed * 0.02 * np.sin(2 * np.pi * 0.1 * time)
        total_displacement = displacement + wind_effect
        displacements.append(total_displacement)
    
    return displacements

if __name__ == "__main__":
    # Example usage
    beam_length = 384
    steps = 5
    sway = simulate_crane_sway(beam_length, steps)
    print(f"Crane sway displacements: {sway}")
