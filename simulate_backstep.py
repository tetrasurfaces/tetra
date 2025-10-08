# simulate_backstep.py
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

from gyrogimbal import tilt  # Assume gyrogimbal.py has tilt function for crane sway
from telemetry import log  # Assume telemetry.py has log function

def simulate_backstep(bead_length=9, steps=10, crane_sway=True):
    """
    Simulates backstep welding with manual handling and overhead crane.
    - bead_length: Length of each bead in inches (default 9).
    - steps: Number of backsteps (default 10).
    - crane_sway: Boolean to model crane sway (default True).
    Returns: List of temp readings per step.
    """
    temp_readings = []
    for step in range(steps):
        # Simulate backstep
        back = bead_length / 5  # Back up 20% of bead length
        log(f"Step {step}: Advance {bead_length} in, back {back} in")
        
        # Manual handling with crane
        if crane_sway:
            tilt("crane", degrees=2)  # Simulate 2-degree sway
            log("Crane sway adjusted")
        
        # Temp reading
        temp = 250 - step * 5  # Simulated interpass temp drop
        temp_readings.append(temp)
        log(f"Interpass temp: {temp}C")
    
    return temp_readings

# Example usage
if __name__ == "__main__":
    readings = simulate_backstep()
    print(f"Temp readings: {readings}")
