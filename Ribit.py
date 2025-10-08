# Ribit.py
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import hashlib
from PIL import Image
from green_curve import bspline_basis, custom_interoperations_green_curve
from friction_vibe import friction_vibe

def ribit_generate(data):
    ribit_hash = hashlib.sha256(data.encode()).digest()
    ribit_int = int.from_bytes(ribit_hash, 'big') % (1 << 7)
    state = ribit_int % 7
    rainbow_colors = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet']
    color = rainbow_colors[state]
    return ribit_int, state, color

class TetraRibit:
    def __init__(self):
        self.center = np.array([0, 0, 0])
        self.colored_points = [np.array([ -0.4, -0.2, 0 ]), np.array([ -0.3, -0.3, 0 ]), np.array([ 0.4, -0.3, 0 ]), np.array([ 0.5, 0.1, 0 ]), np.array([ 0.3, 0.3, 0 ]), np.array([ -0.2, 0.2, 0 ])]
        self.colors = ['orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
        self.height = 0.5
        self.num_layers = 50
        self.z_levels = np.linspace(0, self.height, self.num_layers)
        self.arms_3d = []

    def gyro_gimbal(self, pos1, pos2, tilt=np.array([0.1,0.1,0.1]), kappa=0.3):
        dist = np.linalg.norm(pos1 - pos2)
        if dist < 0.1:
            vibe, base_gyro = self.friction_vibe(pos1, pos2, kappa)
            gimbal_spin = base_gyro + tilt / dist if dist > 0 else base_gyro
            warp = 1 / (1 + kappa * dist)
            return vibe * warp, gimbal_spin
        else:
            return 0, np.zeros(3)

    def generate_arms(self):
        for i, point in enumerate(self.colored_points):
            ribit_int, state, color = ribit_generate('arm' + str(i))
            kappa_mid = 1.0 + state / 7.0
            mid_point = (self.center[:2] + point[:2]) / 2 + 0.05
            arm_points = [self.center[:2], mid_point, point[:2]]
            kappas = [1.0, kappa_mid, 1.0]
            smooth_x, smooth_y = custom_interoperations_green_curve(arm_points, kappas, is_closed=False)
            arm_3d = np.zeros((len(smooth_x), 3))
            arm_3d[:, 0] = smooth_x
            arm_3d[:, 1] = smooth_y
            self.arms_3d.append(arm_3d)
            # Vibe on segments
            for j in range(1, len(arm_3d)):
                pos1 = arm_3d[j-1]
                pos2 = arm_3d[j]
                wave, spin = self.gyro_gimbal(pos1, pos2)
                print(f"Arm {i} seg {j}: Wave {wave}, Spin {spin}")

    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, arm_3d in enumerate(self.arms_3d):
            for z in self.z_levels:
                arm_z = arm_3d.copy()
                arm_z[:, 2] = z
                ax.plot(arm_z[:, 0], arm_z[:, 1], arm_z[:, 2], color=self.colors[i])
            ax.scatter(arm_3d[-1, 0], arm_3d[-1, 1], self.height, color=self.colors[i], s=50)
        ax.scatter(self.center[0], self.center[1], self.height, color='red', s=100)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Ribit with Green Curves')
        png_filename = 'ribit_structure.png'
        plt.savefig(png_filename, dpi=300)
        print(f"Saved {png_filename}")
        self.raster_to_light(png_filename)
        plt.show()

    def raster_to_light(self, png_file):
        img = Image.open(png_file)
        pixels = np.array(img)
        light_hash = hashlib.sha256(pixels.tobytes()).hexdigest()[:16]
        print(f"Light hash: {light_hash}")
        intensity = int(light_hash, 16) % 256
        print(f"Intensity: {intensity}")

if __name__ == "__main__":
    telemetry = TetraRibit()
    telemetry.generate_arms()
    telemetry.visualize()
