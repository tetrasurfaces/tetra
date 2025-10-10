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

class Crane:
    """Simulate crane dynamics with control and stability features."""
    def __init__(self, beam_length, load_weight=1000.0, damping=0.1):
        self.beam_length = beam_length
        self.load_weight = load_weight  # in kg
        self.damping = damping
        self.motor_speed = 0.0  # Initial motor speed (m/s)
        self.control_mode = "manual"  # Default control mode

    def apply_control(self, motor_speed, mode="manual"):
        """Apply motor control and switch mode."""
        self.motor_speed = np.clip(motor_speed, -5.0, 5.0)  # Limit motor speed
        self.control_mode = mode
        print(f"Applied control: Motor speed = {self.motor_speed} m/s, Mode = {self.control_mode}")

    def simulate_crane_sway(self, steps, wind_speed=5.0):
        """
        Simulate crane sway displacement based on beam length, load, and control.
        
        Args:
            steps (int): Number of simulation steps.
            wind_speed (float, optional): Wind speed in m/s. Defaults to 5.0.
        
        Returns:
            list: List of displacement values (in meters) for each step.
        """
        amplitude = 0.1 * self.beam_length + 0.001 * self.load_weight  # Load increases sway
        frequency = 0.5 / self.beam_length  # Frequency inversely proportional to length
        control_factor = 0.1 * self.motor_speed if self.control_mode == "manual" else 0.05 * self.motor_speed  # Automated reduces sway
        
        displacements = []
        for i in range(steps):
            time = i * 0.1  # Time step of 0.1 seconds
            # Harmonic oscillation with damping, load, wind, and control effects
            displacement = amplitude * np.sin(2 * np.pi * frequency * time) * np.exp(-self.damping * time)
            wind_effect = wind_speed * 0.02 * np.sin(2 * np.pi * 0.1 * time)
            control_effect = control_factor * np.cos(2 * np.pi * 0.2 * time)  # Oscillatory control
            total_displacement = displacement + wind_effect + control_effect
            displacements.append(total_displacement)
        
        return displacements

    def get_stability(self):
        """Assess crane stability based on load and sway amplitude."""
        max_sway = 0.1 * self.beam_length + 0.001 * self.load_weight
        stability = 1.0 / (1.0 + max_sway * self.damping)
        return np.clip(stability, 0.0, 1.0)

    def switch_control_mode(self, mode):
        """Switch between manual and automated control modes."""
        if mode in ["manual", "auto"]:
            self.control_mode = mode
            print(f"Switched to {mode} mode")
        else:
            print(f"Invalid mode: {mode}. Use 'manual' or 'auto'")

if __name__ == "__main__":
    # Example usage
    crane = Crane(beam_length=384, load_weight=2000.0)
    crane.apply_control(motor_speed=2.0, mode="auto")
    sway = crane.simulate_crane_sway(steps=5)
    print(f"Crane sway displacements: {sway}")
    stability = crane.get_stability()
    print(f"Crane stability: {stability:.2f}")
    crane.switch_control_mode("manual")
