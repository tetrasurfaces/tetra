# ribit_telemetry.py
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import hashlib
from PIL import Image
from wise_transforms import bitwise_transform, hexwise_transform, hashwise_transform
from id_util_nurks_closure_hex import custom_interoperations_green_curve, bspline_basis
# Mock for runnable

# From friction_vibe / gyro_gimbal class TetraVibe
class TetraVibe:
    def friction_vibe(self, pos1, pos2, kappa=0.3):
        dist = np.linalg.norm(pos1 - pos2)
        if dist < 1e-6:
            print("heat spike-flinch")
            return 1.0, np.zeros(3)
        if dist < 0.1:
            vibe = np.sin(2 * np.pi * dist / 0.05)
            gyro = np.cross(pos1, pos2) / dist
            warp = 1 / (1 + kappa * dist)
            return vibe * warp, gyro
        else:
            return 1.0, np.zeros(3)

    def gyro_gimbal_rotate(self, coords, angles):
        rot_x = np.array([[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]])
        rot_y = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        rot_z = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]])
        rot = rot_z @ rot_y @ rot_x
        return np.dot(coords, rot.T)

class RibitTelemetry:
    def __init__(self, coords, entropies):
        self.coords = coords  # Accum from kappa
        self.entropies = entropies
        self.center = np.array([0, 0, 0])
        self.colors = ['orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
        self.height = 0.5
        self.num_layers = 50
        self.vibe_model = TetraVibe()

    def generate(self):
        arms_3d = []
        for i, (coord, ent) in enumerate(zip(self.coords, self.entropies)):
            pos = np.array(coord)
            mid_point = (self.center + pos) / 2 + 0.05 * np.random.randn(3)
            arm_points = [self.center, mid_point, pos]
            smooth_x, smooth_y = custom_interoperations_green_curve(arm_points, [1.0]*3, False)
            arm_3d = np.zeros((len(smooth_x), 3))
            arm_3d[:, 0] = smooth_x
            arm_3d[:, 1] = smooth_y
            # Foam if high ent
            if ent > 100:
                arm_3d[:, 2] += np.random.uniform(0, 0.1, len(smooth_x))  # Foam extrude
            arms_3d.append(arm_3d)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, arm_3d in enumerate(arms_3d):
            prev_pos = self.center
            for z in np.linspace(0, self.height, self.num_layers):
                arm_z = arm_3d.copy()
                arm_z[:, 2] = z
                vibe, _ = self.vibe_model.friction_vibe(prev_pos, arm_z[0])
                z_levels = np.linspace(0, self.height, self.num_layers) * vibe  # Warp
                angles = np.array([0.1, 0.2, 0.3])  # From spin
                arm_z = self.vibe_model.gyro_gimbal_rotate(arm_z, angles)
                ax.plot(arm_z[:, 0], arm_z[:, 1], arm_z[:, 2], color=self.colors[i % len(self.colors)])
                prev_pos = arm_z[-1]
            ax.scatter(arm_3d[-1, 0], arm_3d[-1, 1], self.height, color=self.colors[i % len(self.colors)], s=50)
        ax.scatter(self.center[0], self.center[1], self.height, color='red', s=100)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Ribit Structure with Green Curves')
        png_filename = 'ribit_structure.png'
        plt.savefig(png_filename, dpi=300)
        print(f"Saved ribit structure as PNG: {png_filename}")
        self.raster_to_light(png_filename)
        plt.show()
        print("The 3D ribit structure is generated with smoothed green curves for each arm, extruded along Z for depth. The center is at (0,0) with colored points at petal ends.")

    def raster_to_light(self, png_file):
        img = Image.open(png_file)
        pixels = np.array(img)
        salted_bytes = secure_hash_two(pixels.tobytes().hex(), 'she_key', 'foam')  # Secure she salt
        light_hash = hashlib.sha256(salted_bytes.encode()).hexdigest()[:16]
        bit = bitwise_transform(light_hash)
        hex_out = hexwise_transform(light_hash)
        hash_out, ent = hashwise_transform(light_hash)
        hybrid = f"{bit}:{hex_out}:{hash_out}"
        print(f"Rasterized to light hybrid: {hybrid}")
        intensity = int(light_hash, 16) % 256
        print(f"Light intensity: {intensity} (sim GPIO PWM)")
        return hybrid

# Mock coords ents for test
if __name__ == "__main__":
    coords = [ [0.4, 0.2, 0.1], [-0.3, -0.3, 0.2], [0.4, -0.3, 0.3], [0.5, 0.1, 0.4], [0.3, 0.3, 0.5], [-0.2, 0.2, 0.6] ]
    entropies = [50, 150, 80, 200, 90, 120]
    ribit = RibitTelemetry(coords, entropies)
    ribit.generate()
