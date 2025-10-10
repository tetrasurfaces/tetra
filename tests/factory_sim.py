# factory_sim.py
# Copyright 2025 Beau Ayres
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
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

import simpy
import numpy as np
import hashlib
import time
from ghost_hand import Ghosthand

class FactorySim:
    def __init__(self, env):
        self.env = env
        self.gate = np.array([0, 0, 0])
        self.kappa = 0.1  # baseline curvature
        self.history = []  # kappa register
        self.lockouts = set()  # lockout tags
        self.sensors = []  # two-camera array placeholder
        self.ghosthand = GhostHand()  # haptic feedback

    def register_kappa(self, incident=None):
        now = time.time()
        key = hashlib.sha3_256(f"{now}{self.kappa:.2f}{incident or ''}".encode()).hexdigest()
        self.history.append((now, self.kappa, key))
        if len(self.history) > 1000:
            self.history.pop(0)  # keep last 1000
        return key

    def get_situational_kappa(self):
        if not self.history:
            return self.kappa
        last_kappa = self.history[-1][1]
        if len(self.history) > 5:
            drift = np.std([k[1] for k in self.history[-5:]])
            return last_kappa + drift  # real-time awareness
        return last_kappa

    def trigger_emergency(self, incident):
        self.kappa += 0.2  # reflex to hazard
        self.lockouts.add(incident)
        self.ghosthand.pulse(2)  # alert workers
        print(f"{incident.upper()} - Kappa now {self.kappa:.2f}")

    def auto_rig(self, target, repair_time=5):
        yield self.env.timeout(repair_time)
        print(f"{target} fixed by drone. Lockout cleared.")
        self.lockouts.discard(target)
        self.kappa -= 0.2  # settle
        self.ghosthand.pulse(1)  # clear signal

    def camera_array(self):
        # two-camera stereo, 15in baseline
        points = np.random.rand(100, 3) * 100  # mock point cloud
        drift = np.linalg.norm(points[-1] - self.gate)
        if drift > 5:  # threshold for path deviation
            self.kappa += 0.05
            self.ghosthand.pulse(3)  # warn of drift
        return points

    def run_day(self):
        print(f"Day start - Situational Kappa = {self.get_situational_kappa():.3f}")
        yield self.env.timeout(20)  # 20s to rupture
        self.trigger_emergency("gas_rupture")
        self.register_kappa("gas_rupture")
        yield self.env.process(self.auto_rig("gas_line"))
        self.camera_array()  # update kappa
        self.register_kappa()
        print(f"Day end - Situational Kappa = {self.get_situational_kappa():.3f}")

if __name__ == "__main__":
    env = simpy.Environment()
    sim = FactorySim(env)
    env.process(sim.run_day())
    env.run(until=60)
