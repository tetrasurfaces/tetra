# arc_control.py
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
# kappasha_os.py - Kappa-tilted OS with rhombus voxel navigation, factory sim integration.
# CLI-driven, no GUI, DOS Navigator soul in 3D. Pure civilian engineering.

import simpy
import numpy as np
from nav3d import RhombusNav
from factory_sim import FactorySim
from ghost_hand import GhostHand
from thought_curve import ThoughtCurve
from arch_utils.render import render
from dev_utils.lockout import lockout
from dev_utils.hedge import hedge, multi_hedge
from dev_utils.grep import grep
from dev_utils.thought_arb import thought_arb
import kappasha_os_cython  # Cython-optimized functions

class KappashaOS:
    def __init__(self):
        self.env = simpy.Environment()
        self.nav = RhombusNav(kappa=0.2)
        self.factory = FactorySim(self.env)
        self.hand = GhostHand(kappa=0.2)
        self.curve = ThoughtCurve()
        self.commands = []
        self.sensor_data = []
        self.decisions = []  # Log decision outcomes
        print("Kappasha OS booted - kappa-tilted rhombus grid with decision support ready.")

    def poll_sensor(self):
        """Simulate real-time sensor input (gyro, camera)."""
        while True:
            gyro = np.random.uniform(0, 20)
            drift = np.random.rand() * 0.1
            self.sensor_data.append((self.env.now, gyro, drift))
            if gyro > 10 or drift > 0.05:
                self.nav.kappa += 0.1
                self.factory.kappa += 0.1
                self.hand.kappa += 0.1
                self.hand.pulse(2)
                print(f"Sensor alert: Kappa adjusted to {self.nav.kappa:.3f} (gyro={gyro:.1f}, drift={drift:.2f})")
            yield self.env.timeout(5)

    def run_command(self, cmd):
        """Execute CLI commands with kappa awareness."""
        self.commands.append(cmd)
        if cmd == "kappa ls":
            front, right, top = kappasha_os_cython.project_third_angle(self.nav.grid, self.nav.kappa)  # Cython call
            print("FRONT:\n", front[:3, :3])
            print("RIGHT:\n", right[:3, :3])
            print("TOP:\n", top[:3, :3])
        elif cmd.startswith("kappa tilt"):
            try:
                dk = float(cmd.split()[2])
                self.nav.kappa += dk
                self.factory.kappa += dk
                self.hand.kappa += dk
                self.hand.pulse(2)
                print(f"Kappa now {self.nav.kappa:.3f}")
            except:
                print("usage: kappa tilt 0.05")
        elif cmd.startswith("kappa cd"):
            try:
                path = cmd.split()[2]
                self.nav.path.append(path)
                hedge_action = hedge(self.curve, self.nav.path)
                if hedge_action == "unwind":
                    self.hand.pulse(3)
                    print("Path hedge: unwind")
                print(f"Curved to /{path}")
            except:
                print("usage: kappa cd logs")
        elif cmd.startswith("kappa unlock"):
            try:
                coord = tuple(map(int, cmd.split()[2].strip("()").split(",")))
                if self.nav.unlock_edge(coord):
                    self.factory.register_kappa("edge_unlock")
            except:
                print("usage: kappa unlock (7,0,0)")
        elif cmd == "arch_utils render":
            filename = render(self.nav.grid, self.nav.kappa)
            print(f"arch_utils: Rendered to {filename}")
        elif cmd.startswith("dev_utils lockout"):
            try:
                target = cmd.split()[2]
                lockout(self.factory, target)
            except:
                print("usage: dev_utils lockout gas_line")
        elif cmd.startswith("kappa grep"):
            try:
                pattern = cmd.split(maxsplit=2)[2]
                matches = grep(self.factory.history, pattern)
                if matches:
                    self.hand.pulse(len(matches))
                    print(f"Grep found {len(matches)} matches:")
                    for m in matches[:3]:
                        print(f" - {m}")
                else:
                    print("No matches found.")
            except:
                print("usage: kappa grep /warp=0.2+/")
        elif cmd == "kappa sensor":
            print(f"Sensor data: {self.sensor_data[-1]}")
        elif cmd.startswith("kappa hedge multi"):
            try:
                paths = cmd.split()[2].strip("[]").split(",")
                paths = [p.strip() for p in paths]
                hedge_action = multi_hedge(self.curve, [(paths[-2], paths[-1])] if len(paths) > 1 else [(paths[0], paths[0])])
                if "unwind" in hedge_action:
                    self.hand.pulse(4)
                    print(f"Multi-path hedge: {hedge_action}")
                else:
                    print(f"Multi-path hedge: {hedge_action}")
            except:
                print("usage: kappa hedge multi [gate,weld]")
        elif cmd.startswith("kappa decide"):
            try:
                intent = cmd.split()[2]
                action = kappasha_os_cython.thought_arb_cython(self.curve, self.factory.history, intent)  # Cython call
                self.decisions.append((self.env.now, intent, action))
                self.hand.pulse(2 if action == "unwind" else 1)
                print(f"Decision: {intent} - {action}")
                if action == "unwind":
                    self.nav.kappa += 0.05
                    print(f"Kappa adjusted to {self.nav.kappa:.3f} due to arbitrage")
            except:
                print("usage: kappa decide weld")
        else:
            print("kappa: ls | tilt 0.05 | cd logs | unlock (7,0,0) | arch_utils render | dev_utils lockout gas_line | grep /warp=0.2+/ | sensor | hedge multi [gate,weld] | decide weld")

    def run_day(self):
        """Simulate a factory day with kappa navigation."""
        print(f"Day start - Situational Kappa = {self.factory.get_situational_kappa():.3f}")
        self.env.process(self.poll_sensor())
        yield self.env.timeout(20)
        self.factory.trigger_emergency("gas_rupture")
        self.factory.register_kappa("gas_rupture")
        self.run_command("kappa cd weld")
        self.run_command("kappa unlock (7,0,0)")
        self.run_command("kappa grep /gas_rupture/")
        self.run_command("kappa sensor")
        self.run_command("kappa hedge multi [gate,weld]")
        self.run_command("kappa decide weld")
        yield self.env.process(self.factory.auto_rig("gas_line"))
        self.run_command("kappa ls")
        self.run_command("arch_utils render")
        print(f"Day end - Situational Kappa = {self.factory.get_situational_kappa():.3f}")
        print(f"Decisions made: {self.decisions}")

if __name__ == "__main__":
    os = KappashaOS()
    os.env.process(os.run_day())
    os.env.run(until=60)
