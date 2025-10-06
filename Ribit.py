# Copyright Todd Hutchinson, Beau Ayres, Anonymous
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
# express written permission from Todd Hutchinson and Beau Ayres.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import hashlib
from PIL import Image

def bspline_basis(u, i, p, knots):
    if p == 0:
        return 1.0 if knots[i] <= u < knots[i + 1] else 0.0
    term1 = ((u - knots[i]) / (knots[i + p] - knots[i])) * bspline_basis(u, i, p - 1, knots) if (knots[i + p] - knots[i]) > 0 else 0
    term2 = ((knots[i + p + 1] - u) / (knots[i + p + 1] - knots[i + 1])) * bspline_basis(u, i + 1, p - 1, knots) if (knots[i + p + 1] - knots[i + 1]) > 0 else 0
    return term1 + term2

def custom_interoperations_green_curve(points, kappas, is_closed=False):
    points = np.array(points)
    kappas = np.array(kappas)
    degree = 3
    num_output_points = 1000
    n = len(points)
    if is_closed:
        knots = np.concatenate((np.arange(-degree, 0), np.linspace(0, n, n - degree + 2), np.arange(n, n + degree)))
    else:
        knots = np.concatenate((np.zeros(degree + 1), np.linspace(0, 1, n - degree + 1)[1:-1], np.ones(degree + 1)))
    u_fine = np.linspace(0, 1, num_output_points)
    smooth_x = np.zeros(num_output_points)
    smooth_y = np.zeros(num_output_points)
    for j, u in enumerate(u_fine):
        num_x, num_y, den = 0.0, 0.0, 0.0
        for i in range(n):
            b = bspline_basis(u, i, degree, knots)
            w = kappas[i] * b
            num_x += w * points[i, 0]
            num_y += w * points[i, 1]
            den += w
        if den > 0:
            smooth_x[j] = num_x / den
            smooth_y[j] = num_y / den
    return smooth_x, smooth_y

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

    def friction_vibe(self, pos1, pos2, kappa=0.3):
        dist = np.linalg.norm(pos1 - pos2)
        if dist < 0.1:
            vibe = np.sin(2 * np.pi * dist / 0.05)
            gyro = np.cross(pos1, pos2) / dist if dist > 0 else np.zeros(3)
            warp = 1 / (1 + kappa * dist)
            return vibe * warp, gyro
        else:
            return 0, np.zeros(3)

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
