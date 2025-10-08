# seeing_layer.py
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

def simulate_seeing_layer(heat_delta=0.5, angle=45, air_temp=30):
    """
    Simulates hot air jet for predictable atmospheric blur in range finding.
    - heat_delta: Temperature increase in Kelvin (default 0.5).
    - angle: Jet angle in degrees (default 45).
    - air_temp: Ambient air temperature in Celsius (default 30).
    Returns: Blur radius in mm.
    """
    # Simplified turbulence model for seeing blur
    blur_radius = heat_delta * np.sin(np.radians(angle)) * (air_temp / 30)
    rig = Rig()
    rig.log("Seeing layer activated", heat_delta=heat_delta, angle=angle, blur=blur_radius)
    print(f"Seeing layer blur radius: {blur_radius:.2f} mm")
    return blur_radius

# Example usage
if __name__ == "__main__":
    blur = simulate_seeing_layer(heat_delta=0.5, angle=45)
