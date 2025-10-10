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
# plasma.py - Plasma cutting control with arc monitoring.
# Integrates with KappashaOS for real-time feedback.

import time
from ghost_hand import GhostHand
from arc_listen import ArcListener
from arc_control import ArcControl

class PlasmaCutter:
    def __init__(self):
        self.hand = GhostHand(kappa=0.2)
        self.listener = ArcListener()
        self.control = ArcControl(set_voltage=220.0, set_feed_rate=50.0)  # Plasma-specific
        self.log = []
        self.running = True
        self.z_offset = 0.0  # Z-axis drift in mm

    def cut(self):
        """Simulate plasma cutting with arc and Z-axis monitoring."""
        while self.running:
            # Simulate sensor data (replace with hardware)
            current_voltage = self.control.voltage + (time.time() % 2 - 1) * 5.0  # ±5V drift
            self.z_offset = (time.time() % 10 - 5) * 0.2  # ±1mm drift
            sound_profile = self.listener.analyze_sound(self.listener.stream.read(self.listener.CHUNK, exception_on_overflow=False))

            # Log data
            self.log.append((time.time(), current_voltage, self.z_offset, sound_profile["frequency"]))

            # Arc feedback
            if sound_profile["frequency"] > 8000:  # Kerf instability or spatter
                self.hand.pulse(2)
                print("Kerf wobble detected - adjust speed or Z")
            elif abs(self.z_offset) > 2.0:  # Z-axis drift
                self.hand.pulse(1)
                print(f"Z-axis drift {self.z_offset:.1f}mm - recalibrate")

            # Safety cutoff
            if current_voltage > self.control.voltage + 5.0:  # 5V over target
                print("Voltage spike detected - shutting down")
                self.running = False

            time.sleep(0.1)  # 10ms loop

    def stop(self):
        """Stop cutting and clean up."""
        self.running = False
        self.listener.stop()
        print("Plasma Cutter Log:", self.log[-5:])  # Last 5 entries

if __name__ == "__main__":
    cutter = PlasmaCutter()
    try:
        cutter.cut()
    except KeyboardInterrupt:
        cutter.stop()
