# particles.py
# Copyright 2025 Beau Ayres
# Proprietary Software - All Rights Reserved
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
from frictionvibe import track_kappa_vector  # Import from frictionvibe.py
from telemetry import log  # Assume telemetry.py has log function

def track_particle_vector(stages, drag=0.05):
    """
    Tracks particle vectors from forge to weld in the supply chain.
    - stages: List of (x, y, z, stage_name) tuples (e.g., 'forge', 'ship', 'weld').
    - drag: Drag coefficient (default 0.05).
    Returns: List of particle vectors.
    """
    vectors = track_kappa_vector(stages, drag)  # Reuse kappa vector logic
    for i, (vector, stage) in enumerate(zip(vectors, [s[3] for s in stages])):
        log(f"Particle {i}: Vector {vector}, Stage {stage}")
    
    return vectors

# Example usage
if __name__ == "__main__":
    stages = [(0, 0, 0, 'forge'), (10, 5, 0, 'ship'), (15, 5, 2, 'weld')]
    particle_vectors = track_particle_vector(stages)
    print(f"Particle vectors: {particle_vectors}")
