# gyro_gimbal.py
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
import mpmath
mpmath.mp.dps = 19
from tetra.utils.periodic_table import Element  # For space_gravity_constant

class Sym:
    """Gyroscopic rig for stabilizing and tilting atomic or rig simulations."""
    def __init__(self):
        self.spin_rate = 0.0
        self.tilt_angle = np.array([0.0, 0.0, 0.0])
    
    def tilt(self, axis, rate):
        """Tilt the rig or simulate atomic spin along a given axis."""
        if axis == "spin_axis":
            self.spin_rate = rate
            print(f"Tilting spin axis at rate {rate}")
        else:
            idx = {"x": 0, "y": 1, "z": 2}.get(axis[0].lower(), 0)
            self.tilt_angle[idx] = rate
            print(f"Tilting {axis} by {rate} degrees")
    
    def stabilize(self):
        """Stabilize the rig or atomic structure after tilting."""
        self.spin_rate = 0.0 if abs(self.spin_rate) < 1e-6 else self.spin_rate * 0.9  # Damping
        self.tilt_angle = np.where(abs(self.tilt_angle) < 1e-6, 0.0, self.tilt_angle * 0.9)
        print("Stabilizing gyro, spin rate reduced to", self.spin_rate)
    
    def get_spin_vector(self):
        """Return the current 3D spin vector based on spin_rate and tilt_angle."""
        # Combine spin_rate (scalar) with tilt_angle (vector) to form a 3D spin vector
        spin_magnitude = self.spin_rate
        spin_direction = self.tilt_angle / np.linalg.norm(self.tilt_angle) if np.linalg.norm(self.tilt_angle) > 0 else np.array([1.0, 0.0, 0.0])
        spin_vector = spin_magnitude * spin_direction
        return spin_vector

class TetraVibe:
    """Class for modeling vibrational and gyroscopic effects at atomic scales."""
    def __init__(self):
        self.gravity_constant = 6.67430e-11  # m³ kg⁻¹ s⁻², from periodic_table.py
    
    def friction_vibe(self, pos1, pos2, kappa=0.3, friction_coeff=0.1):
        """Calculate vibrational and gyroscopic effects with friction."""
        dist = np.linalg.norm(pos1 - pos2)
        if dist < 1e-6:
            print("heat spike-flinch")
            return 1.0, np.zeros(3), 0.0
        if dist < 0.1:
            vibe = np.sin(2 * np.pi * dist / 0.05)
            gyro = np.cross(pos1, pos2) / dist if dist > 0 else np.zeros(3)
            warp = 1 / (1 + kappa * dist)
            # Friction force (proportional to distance and coefficient)
            friction_force = friction_coeff * (1 / dist) if dist > 0 else 0.0
            return vibe * warp, gyro, friction_force
        return 1.0, np.zeros(3), 0.0
    
    def gyro_gimbal(self, pos1, pos2, tilt=np.array([0.1, 0.1, 0.1]), kappa=0.3, element=None, spin_rate=1.0):
        """Simulate gyroscopic gimbal with space properties."""
        dist = np.linalg.norm(pos1 - pos2)
        if dist < 1e-6:
            print("heat spike-flinch")
            return 1.0, np.zeros(3), 0.0
        if dist < 0.1:
            vibe, base_gyro, friction_force = self.friction_vibe(pos1, pos2, kappa)
            gimbal_spin = base_gyro + tilt / dist
            warp = 1 / (1 + kappa * dist)
            # Gravitational force if element data is provided
            grav_force = 0.0
            if isinstance(element, Element):
                mass_kg = 1e-26 * element.atomic_weight  # Approximate atomic mass in kg
                grav_force = self.gravity_constant * (mass_kg ** 2) / (dist * 1e-12) ** 2 if dist > 0 and element.atomic_radius else 0.0
                gimbal_spin += np.array([spin_rate, 0.0, 0.0]) / dist  # Add spin effect
            return vibe * warp, gimbal_spin, friction_force + grav_force
        return 1.0, np.zeros(3), 0.0
    
    def gyro_gimbal_rotate(self, coords, angles=None):
        """Rotate coordinates using gyroscopic angles."""
        if angles is None:
            angles = np.array([float(mpmath.phi), 0.0, 0.0])  # Default phi x
        if len(angles) != 3:
            print("heat spike-flinch")  # Wrong dim
            return coords
        rot_x = np.array([[1, 0, 0],
                          [0, float(mpmath.cos(angles[0])), float(-mpmath.sin(angles[0]))],
                          [0, float(mpmath.sin(angles[0])), float(mpmath.cos(angles[0]))]])
        rot_y = np.array([[float(mpmath.cos(angles[1])), 0, float(mpmath.sin(angles[1]))],
                          [0, 1, 0],
                          [float(-mpmath.sin(angles[1])), 0, float(mpmath.cos(angles[1]))]])
        rot_z = np.array([[float(mpmath.cos(angles[2])), float(-mpmath.sin(angles[2])), 0],
                          [float(mpmath.sin(angles[2])), float(mpmath.cos(angles[2])), 0],
                          [0, 0, 1]])
        rot = rot_z @ rot_y @ rot_x
        det = np.linalg.det(rot)
        if abs(det - 1) > 1e-6:
            print("heat spike-flinch")  # Singular
        return np.dot(coords, rot.T)

if __name__ == "__main__":
    sym = Sym()
    model = TetraVibe()
    pos1 = np.array([0, 0, 0])
    pos2 = np.array([0.05, 0, 0])
    from tetra.utils.periodic_table import carbon
    wave, spin, force = model.gyro_gimbal(pos1, pos2, element=carbon, spin_rate=1.5)
    print(f"Wave: {wave}, Spin: {spin}, Force: {force:.4e}")
    sym.tilt("spin_axis", 1.5)
    sym.stabilize()
    spin_vector = sym.get_spin_vector()
    print(f"Spin Vector: {spin_vector}")
    pos3 = np.array([0.15, 0, 0])  # far fake
    wave_far, spin_far, force_far = model.gyro_gimbal(pos1, pos3)
    print(f"Far wave: {wave_far}, Far spin: {spin_far}, Far force: {force_far:.4e}")
    coord = np.array([[1.0, 0.0, 0.0]])
    angles = np.array([np.pi/2, 0.0, 0.0])
    new_coord = model.gyro_gimbal_rotate(coord, angles)
    print(f"Rotated: {new_coord}")
