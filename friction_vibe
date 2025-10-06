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

class TetraVibe:
    def friction_vibe(self, pos1, pos2, kappa=0.3):
        dist = np.linalg.norm(pos1 - pos2)
        if dist < 0.1:
            vibe = np.sin(2 * np.pi * dist / 0.05)
            gyro = np.cross(pos1, pos2) / dist
            warp = 1 / (1 + kappa * dist)
            return vibe * warp, gyro
        else:
            return 0, np.zeros(3)

    def gyro_gimbal(self, pos1, pos2, tilt=np.array([0.1,0.1,0.1]), kappa=0.3):
        dist = np.linalg.norm(pos1 - pos2)
        if dist < 0.1:
            vibe, base_gyro = self.friction_vibe(pos1, pos2, kappa)
            gimbal_spin = base_gyro + tilt / dist
            warp = 1 / (1 + kappa * dist)
            return vibe * warp, gimbal_spin
        else:
            return 0, np.zeros(3)

if __name__ == "__main__":
    model = TetraVibe()
    pos1 = np.array([0,0,0])
    pos2 = np.array([0.05,0,0])
    wave, spin = model.gyro_gimbal(pos1, pos2)
    print(f"Wave: {wave}, Spin: {spin}")
    pos3 = np.array([0.15,0,0])  # far fake
    wave_far, spin_far = model.gyro_gimbal(pos1, pos3)
    print(f"Far wave: {wave_far}, Far spin: {spin_far}")
