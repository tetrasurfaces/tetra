# solvents.py
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
from centrifuge import simulate_centrifuge_emulsification

def simulate_two_pack_paint(resin_vol=0.5, hardener_vol=0.5, solvent_frac=0.2, evap_rate=0.01, time_step=0.1, steps=300):
    """
    Simulates two-pack paint mixing with solvent evaporation.
    - resin_vol: Volume of resin in liters (default 0.5).
    - hardener_vol: Volume of hardener in liters (default 0.5).
    - solvent_frac: Initial solvent fraction (default 0.2).
    - evap_rate: Solvent evaporation rate per step (default 0.01).
    - time_step: Simulation time step in seconds (default 0.1).
    - steps: Number of simulation steps (default 300).
    Returns: Viscosity profile in Pa·s and remaining solvent fraction.
    """
    rig = Rig()
    viscosity = 0.001  # Initial viscosity (Pa·s)
    solvent = solvent_frac
    viscosity_profile = []
    
    # Run centrifuge emulsification for mixing
    distances = simulate_centrifuge_emulsification(viscosity=viscosity)
    
    for step in range(steps):
        # Increase viscosity as solvent evaporates
        solvent -= evap_rate * time_step
        solvent = max(0, solvent)  # Prevent negative solvent
        viscosity += 0.0001 * (1 - solvent)  # Viscosity increases as solvent decreases
        viscosity_profile.append(viscosity)
        rig.log("Two-pack paint mixing", step=step, viscosity=viscosity, solvent=solvent, emulsion_distance=distances[step])
    
    return viscosity_profile, solvent

# Example usage
if __name__ == "__main__":
    viscosity, solvent = simulate_two_pack_paint()
    print(f"Viscosity profile: {viscosity[:5]}... Remaining solvent: {solvent}")
