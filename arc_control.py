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
# arc_control.py - Real-time arc length and stick-out control for MIG welding.
# Integrates with KappashaOS for haptic feedback and sound cues.

import time
from ghost_hand import GhostHand
from arc_listen import listen_arc

class ArcControl:
    def __init__(self, set_voltage=26.0, set_feed_rate=300.0):
        self.voltage = set_voltage  # Target voltage in volts
        self.feed_rate = set_feed_rate  # Target feed rate in IPM
        self.stick_out_target = 15.0  # Target stick-out in mm
        self.hand = GhostHand(kappa=0.2)
        self.log = []
        self.running = True

    def monitor(self):
        """Monitor arc length and stick-out, adjust in real-time."""
        while self.running:
            # Simulate sensor data (replace with actual hardware)
            current_voltage = self.voltage + (time.time() % 2 - 1) * 1.5  # ±1.5V drift
            current_feed = self.feed_rate + (time.time() % 2 - 1) * 20  # ±20 IPM drift
            stick_out = 0.8 * (current_feed / (current_voltage / 10))  # Rough estimate

            # Arc length calculation (simplified: voltage/current ratio)
            arc_length = 0.5 * (current_voltage - self.voltage) + 10  # Baseline 10mm

            # Log data
            self.log.append((time.time(), arc_length, stick_out, current_voltage, current_feed))

            # Adjust based on sound feedback
            sound_profile = listen_arc()
            if sound_profile["frequency"] > 8000:  # Spatter or too long
                self.hand.pulse(2)  # Two pulses: pull back
                print("Arc too long/stuttering - adjust stick-out")
            elif sound_profile["frequency"] < 2000:  # Too short
                self.hand.pulse(1)  # One pulse: push in
                print("Arc too short - adjust stick-out")

            # Safety cutoff
            if current_voltage > self.voltage + 5.0:  # 5V over target
                print("Voltage spike detected - shutting down")
                self.running = False

            # Haptic feedback for stick-out
            if stick_out > self.stick_out_target + 3.0:  # >18mm
                self.hand.squeeze(0.8)  # Gentle squeeze: too far
                print(f"Stick-out {stick_out:.1f}mm - pull back")
            elif stick_out < self.stick_out_target - 3.0:  # <12mm
                self.hand.squeeze(0.4)  # Light squeeze: too close
                print(f"Stick-out {stick_out:.1f}mm - push in")

            time.sleep(0.1)  # 10ms loop

    def stop(self):
        """Stop monitoring and print log."""
        self.running = False
        print("Arc Control Log:", self.log[-5:])  # Last 5 entries

if __name__ == "__main__":
    controller = ArcControl()
    try:
        controller.monitor()
    except KeyboardInterrupt:
        controller.stop()
