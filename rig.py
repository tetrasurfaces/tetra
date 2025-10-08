# rig.py
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

import os
import csv
import hashlib
import numpy as np
from datetime import datetime

class Rig:
    def __init__(self, log_file="weld_log.csv"):
        self.angle = 0
        self.torque = 0
        self.log_file = os.environ.get("TELEMETRY_LOG_FILE", log_file)
    
    def tilt(self, direction, degrees):
        """Adjust torch or jib angle for weave or cut."""
        self.angle += degrees
        self.log(f"Tilted {direction} by {degrees} degrees", angle=self.angle)
        print(f"Tilted {direction} by {degrees} degrees")
    
    def stabilize(self):
        """Stabilize against crane sway or wind."""
        self.torque = 0
        self.log("Stabilized rig", torque=self.torque)
        print("Stabilized rig")
    
    def log(self, event, **kwargs):
        """Log telemetry data to CSV."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {"timestamp": timestamp, "event": event}
        row.update(kwargs)
        file_exists = os.path.exists(self.log_file)
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    
    def flag(self, issue):
        """Flag an issue in telemetry."""
        self.log(f"flag_{issue}")
    
    def rust_probe(self):
        """Simulate rust detection probe."""
        return 0
    
    def depth_error(self):
        """Simulate depth error check."""
        return False
    
    def crack_location(self):
        """Simulate crack location detection."""
        return None
    
    def log_quench(self, temp_profile, porosity_threshold=0.2):
        """Log temperature-drop profile during oil quenching."""
        for t, temp in enumerate(temp_profile):
            voids = (200 - temp) / 200 * porosity_threshold if temp < 200 else 0
            self.log(f"Quench step {t}", temp=temp, void_growth=voids * 100)
        print("Quench log complete")
    
    def log_ipfs_navigation(self, vector_data, cache_moves=5):
        """Log supply chain vector navigation with IPFS-style hashing."""
        ipfs_hashes = []
        for i, (pos, speed, load) in enumerate(vector_data):
            move_str = f"{pos}_{speed}_{load}"
            move_hash = hashlib.sha256(move_str.encode()).hexdigest()
            ipfs_hashes.append(move_hash)
            self.log(f"Move {i}", pos=pos, speed=speed, load=load, hash=move_hash[:8])
            if i >= cache_moves:
                self.log(f"Evict move", hash=ipfs_hashes[i - cache_moves][:8])
                ipfs_hashes.pop(0)
        return ipfs_hashes
    
    def log_voxel_metrics(self, voxel_grid, void_count):
        """Log rhombohedral voxel metrics."""
        void_density = void_count / voxel_grid.size
        self.log("Voxel analysis", void_density=void_density, void_count=void_count)
        print(f"Logged voxel metrics: {void_count} voids, density {void_density:.4f}")

# Example usage
if __name__ == "__main__":
    rig = Rig()
    rig.tilt("left", 20)
    rig.stabilize()
    rig.log("test_event", amps=60, volts=182)
    rig.flag("hydrogen")
    rig.log_quench([900, 700, 500, 300, 100, 20])
    rig.log_ipfs_navigation([(0, 10, 100), (1, 12, 95)], cache_moves=2)
    grid = np.random.rand(10, 10, 10)
    rig.log_voxel_metrics(grid, void_count=50)
