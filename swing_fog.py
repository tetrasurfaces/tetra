# swing_fog.py
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

def model_swing_fog(distance=40, index=1.002, glycol_frac=0.001):
    """
    Models ultrasonic mist for refractive index bending in factory environments.
    - distance: Cannon spacing in meters (default 40).
    - index: Refractive index bump (default 1.002).
    - glycol_frac: Glycol fraction in mist (default 0.001).
    Returns: Bend radius in radians per meter.
    """
    # Simulate mist refraction
    bend_radius = (index - 1) * distance  # Simplified Snell's law approximation
    rig = Rig()
    rig.log("Swing fog activated", distance=distance, index=index, bend=bend_radius)
    print(f"Swing fog bend radius: {bend_radius:.4f} rad/m")
    return bend_radius

# Example usage
if __name__ == "__main__":
    bend = model_swing_fog(distance=40, index=1.002)
