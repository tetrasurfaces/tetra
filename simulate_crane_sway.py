# simulate_crane_sway.py
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
from gyrogimbal import tilt  # Assume gyrogimbal.py has tilt function
from telemetry import log  # Assume telemetry.py has log function

def simulate_crane_sway(beam_length=384, sway_angle=2, steps=10):
    """
    Simulates overhead crane sway during welding.
    - beam_length: Length of the beam in inches (default 384 for 32 feet).
    - sway_angle: Maximum sway angle in degrees (default 2).
    - steps: Number of welding steps (default 10).
    Returns: List of sway displacements.
    """
    sway_displacements = []
    for step in range(steps):
        # Simulate crane sway as sinusoidal oscillation
        sway = np.sin(step * np.pi / steps) * sway_angle
        tilt("crane", degrees=sway)  # Apply sway to crane
        displacement = sway * beam_length / 360  # Simplified displacement in inches
        sway_displacements.append(displacement)
        log(f"Step {step}: Sway {sway:.2f} degrees, Displacement {displacement:.2f} inches")
    
    return sway_displacements

# Example usage
if __name__ == "__main__":
    displacements = simulate_crane_sway()
    print(f"Sway displacements: {displacements}")
