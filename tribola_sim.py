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

#!/usr/bin/env python3
# tribola_sim.py - Simulation of Tribola effect with arc friction and light feedback.
# Integrates with arc_listen for real-time response.

import time
import numpy as np
from ghost_hand import GhostHand
from arc_listen import ArcListener

class TribolaSim:
    def __init__(self):
        self.hand = GhostHand(kappa=0.2)
        self.listener = ArcListener()
        self.log = []
        self.running = True
        self.friction_heat = 0.0  # Simulated heat from arc friction

    def simulate(self):
        """Simulate Tribola effect with friction-to-light transition."""
        while self.running:
            # Simulate friction heat (replace with sensor data)
            self.friction_heat = (time.time() % 10) * 10  # 0-100°C range
            sound_profile = self.listener.analyze_sound(self.listener.stream.read(self.listener.CHUNK, exception_on_overflow=False))

            # Log data
            self.log.append((time.time(), self.friction_heat, sound_profile["frequency"]))

            # Tribola detection: high frequency from friction light
            if sound_profile["frequency"] > 10000 and self.friction_heat > 50.0:  # Tribola glow
                self.hand.pulse(3)
                print("Tribola effect detected - photon burst, adjust arc")
            elif self.friction_heat > 80.0:  # Overheat warning
                self.hand.pulse(2)
                print("Friction heat high - reduce speed")

            time.sleep(0.1)  # 10ms loop

    def stop(self):
        """Stop simulation and clean up."""
        self.running = False
        self.listener.stop()
        print("Tribola Sim Log:", self.log[-5:])  # Last 5 entries

if __name__ == "__main__":
    sim = TribolaSim()
    try:
        sim.simulate()
    except KeyboardInterrupt:
        sim.stop()
