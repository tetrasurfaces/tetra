#!/usr/bin/env python3
# site_kappa.py
# copyright 2025 Beau Ayres, xAI
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

import serial
import time
from tetra.arch_utils import calc_live_kappa, tetra_hash_surface, apply_tetra_etch

def read_sensor_raw(port='/dev/ttyUSB0', baud=9600):
    """Read live IMU/gyro data from construction site sensors."""
    try:
        ser = serial.Serial(port, baud, timeout=1)
        while True:
            line = ser.readline().decode('utf-8').strip()
            yield line  # Format: kappa:0.48,drift:0.02
    except Exception as e:
        print(f"Sensor offline: {e}")
        return

def run_formwork_monitor():
    """Monitor construction site curvature, sync with CATIA."""
    mesh = None  # Loaded from CATIA sync via protobuf
    sensor = read_sensor_raw()
    for raw in sensor:
        if 'kappa:' not in raw:
            continue
        kappa_str = raw.split('kappa:')[1].split(',')[0]
        kappa_val = float(kappa_str)
        delta = calc_live_kappa(mesh, target=0.5)
        if abs(delta) > 0.03:
            print(f"ALERT - curvature drift detected: {delta}, re-tension form!")
        hash_val = tetra_hash_surface(mesh)
        apply_tetra_etch(mesh, depth=0.002, hash_val=hash_val)  # Shallow for rebar tag
        time.sleep(1)

if __name__ == "__main__":
    run_formwork_monitor()
