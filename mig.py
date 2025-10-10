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
# mig.py - MIG welding control with pulsed transfer and sound feedback.
# Integrates with KappashaOS for real-time adjustment.

import time
from ghost_hand import GhostHand
from arc_listen import ArcListener
from arc_control import ArcControl

class MIGWelder:
    def __init__(self):
        self.hand = GhostHand(kappa=0.2)
        self.listener = ArcListener()
        self.control = ArcControl(set_voltage=26.0, set_feed_rate=300.0)  # MIG-specific
        self.log = []
        self.running = True
        self.pulse_on = True  # 5ms pulse cycle
        self.pulse_time = 0.0

    def weld(self):
        """Simulate MIG welding with pulsed transfer and arc monitoring."""
        while self.running:
            current_time = time.time()
            if current_time - self.pulse_time > 0.005:  # 5ms pulse
                self.pulse_on = not self.pulse_on
                self.pulse_time = current_time

            # Simulate sensor data (replace with hardware)
            current_voltage = self.control.voltage + (time.time() % 2 - 1) * 1.5  # ±1.5V drift
            current_feed = self.control.feed_rate + (time.time() % 2 - 1) * 20  # ±20 IPM drift
            sound_profile = self.listener.analyze_sound(self.listener.stream.read(self.listener.CHUNK, exception_on_overflow=False))

            # Log data
            self.log.append((current_time, current_voltage, current_feed, sound_profile["frequency"], self.pulse_on))

            # Arc feedback
            if sound_profile["frequency"] > 8000:  # Spatter or too long
                self.hand.pulse(2)
                print("Spatter/arc too long - reduce stick-out")
            elif sound_profile["frequency"] < 2000:  # Too short
                self.hand.pulse(1)
                print("Arc too short - increase stick-out")

            # Safety cutoff
            if current_voltage > self.control.voltage + 5.0:  # 5V over target
                print("Voltage spike detected - shutting down")
                self.running = False

            time.sleep(0.01)  # 10ms loop

    def stop(self):
        """Stop welding and clean up."""
        self.running = False
        self.listener.stop()
        print("MIG Welder Log:", self.log[-5:])  # Last 5 entries

if __name__ == "__main__":
    welder = MIGWelder()
    try:
        welder.weld()
    except KeyboardInterrupt:
        welder.stop()
