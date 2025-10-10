# electrode.py
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

def simulate_electrode(voltage, amperage, arc_length, electrode_gap):
    """
    Simulate electrode performance for welding.
    
    Args:
        voltage (float): Voltage in volts.
        amperage (float): Amperage in amps.
        arc_length (float): Arc length in mm.
        electrode_gap (float): Electrode gap in mm.
    
    Returns:
        dict: Dictionary containing arc_stability (0-1) and hydrogen_content (ml/100g).
    """
    # Optimal voltage range for stability (e.g., 18-22V for typical welding)
    optimal_voltage = 20.0
    voltage_factor = 1.0 - np.abs(voltage - optimal_voltage) / optimal_voltage
    arc_stability = np.clip(0.5 + 0.5 * voltage_factor - 0.1 * (arc_length / 10.0), 0.0, 1.0)
    
    # Hydrogen content increases with amperage and gap
    base_hydrogen = 1.0  # Base level in ml/100g
    hydrogen_increase = 0.05 * amperage + 0.1 * electrode_gap
    hydrogen_content = np.clip(base_hydrogen + hydrogen_increase, 0.0, 10.0)
    
    return {
        "arc_stability": arc_stability,
        "hydrogen_content": hydrogen_content
    }

if __name__ == "__main__":
    # Example usage
    result = simulate_electrode(voltage=180, amperage=50, arc_length=3, electrode_gap=2)
    print(f"Electrode simulation: Arc Stability = {result['arc_stability']:.2f}, Hydrogen Content = {result['hydrogen_content']:.2f} ml/100g")
