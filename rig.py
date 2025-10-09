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
from tetra.utils.periodic_table import Element  # Import Element base class
from tetra.gyrogimbal import Rig as GyroRig  # Import for gyroscopic modeling

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
        """Stabilize against crane sway or wind, including gyroscopic effects."""
        self.torque = 0
        gyro = GyroRig()
        gyro.stabilize()  # Stabilize using gyro
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
    
    def log_quench(self, temp_profile, porosity_threshold=0.2, temp_threshold=200):
        """Log temperature-drop profile during oil quenching with dynamic porosity."""
        for t, temp in enumerate(temp_profile):
            voids = (temp_threshold - temp) / temp_threshold * porosity_threshold if temp < temp_threshold else 0
            self.log(f"Quench step {t}", temp=temp, void_growth=voids * 100, porosity_threshold=porosity_threshold)
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
    
    def log_mirage(self, heat_temp=900, air_temp=30, bend_radius=0.002):
        """Log and correct for heat-induced mirage effects in range finding."""
        delta_temp = heat_temp - air_temp
        correction = delta_temp * bend_radius  # Simplified correction factor
        self.log("Mirage correction", heat_temp=heat_temp, air_temp=air_temp, correction=correction)
        print(f"Mirage correction applied: {correction:.4f} rad/m")
    
    def log_centrifugal_coriolis(self, centrifugal_force, coriolis_displacement, total_displacement):
        """Log centrifugal and Coriolis effects from centrifuge simulations."""
        self.log("Centrifugal-Coriolis simulation", centrifugal_force=centrifugal_force, coriolis_displacement=coriolis_displacement, total_displacement=total_displacement)
        print(f"Logged centrifugal force: {centrifugal_force:.4f} N, Coriolis displacement: {coriolis_displacement:.4f} m, Total: {total_displacement:.4f} m")
    
    def log_paint_mixing(self, viscosity, solvent_level, emulsion_distance):
        """Log two-pack paint mixing metrics."""
        self.log("Paint mixing", viscosity=viscosity, solvent_level=solvent_level, emulsion_distance=emulsion_distance)
        print(f"Logged paint viscosity: {viscosity:.4f} Pa·s, solvent level: {solvent_level:.4f}, emulsion distance: {emulsion_distance:.4f} m")
    
    def log_element_space_properties(self, element, spin_rate=1.0, friction_coeff=0.1):
        """Log space-related properties of an element using gyroscopic modeling."""
        if isinstance(element, Element):
            structure_map = element.map_structure(spin_rate=spin_rate, friction_coeff=friction_coeff)
            self.log(f"Space properties for {element.name}", 
                     atomic_number=structure_map["atomic_number"],
                     spin_rate=structure_map["spin_rate"],
                     friction_force=structure_map["friction_force"],
                     gravitational_force=structure_map["gravitational_force"])
            print(f"Logged space properties for {element.name}: Spin = {spin_rate}, Friction = {structure_map['friction_force']:.4e} N, Gravity = {structure_map['gravitational_force']:.4e} N")
        else:
            self.log(f"Invalid element: {element}")
            print(f"Invalid element provided: {element}")

# Example usage
if __name__ == "__main__":
    from tetra.utils.periodic_table import carbon, iron
    rig = Rig()
    rig.tilt("left", 20)
    rig.stabilize()
    rig.log("test_event", amps=60, volts=182)
    rig.flag("hydrogen")
    rig.log_quench([900, 700, 500, 300, 100, 20])
    rig.log_ipfs_navigation([(0, 10, 100), (1, 12, 95)], cache_moves=2)
    grid = np.random.rand(10, 10, 10)
    rig.log_voxel_metrics(grid, void_count=50)
    rig.log_mirage(heat_temp=900, air_temp=30)
    rig.log_centrifugal_coriolis(centrifugal_force=0.01, coriolis_displacement=1e-6, total_displacement=0.010001)
    rig.log_paint_mixing(viscosity=0.0015, solvent_level=0.18, emulsion_distance=0.01)
    rig.log_element_space_properties(carbon, spin_rate=1.5, friction_coeff=0.15)
    rig.log_element_space_properties(iron, spin_rate=1.0, friction_coeff=0.1)
