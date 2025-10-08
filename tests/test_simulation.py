# test_simulation.py
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

import pytest
import os
import sys
import numpy as np

# Fallback to add tetra directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tetra import kappa_grid
from porosity_hashing import porosity_hashing
from electrode import simulate_electrode
from crane_sway import simulate_crane_sway
from particle_vector import track_particle_vector
from quantum_sync import quantum_sync
from fleet_vector import simulate_fleet_vector
from tetra.forge_telemetry import log, flag, rust_probe, depth_error, crack_location
from tetra.solid import mesh
from tetra.haptics import buzz, shake
from tetra.welding import weave, TIG, acetylene
from tetra.rig import Rig
from tetra.friction import Friction
from tetra.maptics import Maptics
from tetra.prep_tools import angle_grinder, swarf_vacuum, acetylene_mark, auto_markup
from tetra.test_tools import flex_until_break, ink_test
from tetra.post_process import anodize, viscosity_check, pack, quench, paint

def test_kappa_grid():
    """Test kappa_grid generation."""
    grid = kappa_grid(grid_size=10)
    assert grid.shape == (10, 10, 360), "Unexpected kappa grid shape"

def test_porosity_hashing():
    """Test porosity hashing for void detection."""
    grid = np.random.rand(10, 10, 10)  # Mock grid
    hashed_voids = porosity_hashing(grid, void_threshold=0.3)
    assert isinstance(hashed_voids, dict), "Hashed voids should be a dictionary"
    assert len(hashed_voids) > 0, "No voids detected"

def test_electrode_stability():
    """Test electrode simulation for arc stability and hydrogen content."""
    result = simulate_electrode(voltage=180, amperage=50, arc_length=3, electrode_gap=2)
    assert result['arc_stability'] > 0.7, f"Arc stability too low: {result['arc_stability']}"
    assert result['hydrogen_content'] < 4, f"Hydrogen content too high: {result['hydrogen_content']}"
    assert result['weld_strength'] > 0, "Weld strength should be positive"

def test_crane_sway():
    """Test crane sway simulation for welding."""
    displacements = simulate_crane_sway(beam_length=384, steps=5)
    assert len(displacements) == 5, f"Unexpected number of sway displacements: {len(displacements)}"
    assert all(abs(d) < 10 for d in displacements), "Sway displacement too large"

def test_particle_vector():
    """Test particle vector tracking for supply chain."""
    stages = [(0, 0, 0, 'forge'), (10, 5, 0, 'ship'), (15, 5, 2, 'weld')]
    vectors = track_particle_vector(stages)
    assert len(vectors) == 3, f"Unexpected number of particle vectors: {len(vectors)}"
    assert all(len(v) == 3 for v in vectors), "Invalid vector dimensions"

def test_quantum_sync():
    """Test quantum-inspired synchronization between rigs."""
    rig1 = {'temp': 850, 'crown': 0.1}
    rig2 = {'temp': 849, 'crown': 0.12}
    sync_status = quantum_sync(rig1, rig2, tolerance=0.1)
    assert sync_status, "Rigs should be synchronized within tolerance"

def test_fleet_vector():
    """Test fleet vector simulation for casters."""
    casters = [(1, 0, 10, 100), (2, 1, 12, 95), (3, 2, 11, 98)]
    meta_vec, hashes = simulate_fleet_vector(casters)
    assert isinstance(meta_vec, dict), "Meta-vector should be a dictionary"
    assert len(hashes) <= 5, f"Too many IPFS hashes: {len(hashes)}"
    assert meta_vec['avg_speed'] > 0, "Average speed should be positive"

def test_forge_telemetry_log(tmp_path):
    """Test logging to CSV for welding telemetry."""
    log_file = tmp_path / "weld_log.csv"
    os.environ["TELEMETRY_LOG_FILE"] = str(log_file)
    log("test_event", amps=60, volts=182)
    assert os.path.exists(log_file), "Log file not created"
    with open(log_file, 'r') as f:
        content = f.read()
        assert "test_event" in content, "Event not logged"
        assert "amps" in content, "Amps not logged"

def test_forge_telemetry_flag(tmp_path):
    """Test flagging an issue in telemetry."""
    log_file = tmp_path / "weld_log.csv"
    os.environ["TELEMETRY_LOG_FILE"] = str(log_file)
    flag("hydrogen")
    assert os.path.exists(log_file), "Log file not created"
    with open(log_file, 'r') as f:
        content = f.read()
        assert "flag_hydrogen" in content, "Hydrogen flag not logged"

def test_forge_telemetry_probes():
    """Test telemetry probe functions."""
    assert rust_probe() == 0, "Rust probe should return 0"
    assert depth_error() is False, "Depth error should be False"
    assert crack_location() is None, "Crack location should be None"

def test_solid_mesh():
    """Test mesh generation for welding simulations."""
    mesh_data = mesh("W21x62")
    assert mesh_data["type"] == "W21x62", f"Unexpected mesh type: {mesh_data['type']}"
    assert mesh_data["geometry"] == "hyperbolic_ellipse", f"Unexpected geometry: {mesh_data['geometry']}"

def test_haptics():
    """Test haptic feedback functions."""
    buzz("low")
    shake("hard")
    # Note: Hardware-dependent, assuming no errors raised

def test_welding():
    """Test welding functions for parameter settings."""
    weave("christmas_tree", speed=18, arc=1.8)
    TIG(tungsten="1%lan", argon=98, volts=25, hz=100)
    acetylene(low_oxy=True, duration=0.8)
    # Note: Hardware-dependent, assuming no errors raised

def test_rig():
    """Test rig stabilization and tilt."""
    rig = Rig()
    rig.tilt("left", 20)
    rig.stabilize()
    # Note: Hardware-dependent, assuming stabilization completes

def test_friction():
    """Test friction damping and oscillation."""
    friction = Friction()
    friction.damp(0.5)
    friction.oscillation()
    assert friction.damping == 0.5, f"Unexpected damping value: {friction.damping}"

def test_maptics():
    """Test path recording and replay."""
    maptics = Maptics()
    maptics.record_path(0, 0, 0, 15)
    maptics.replay_path()
    assert len(maptics.path) == 1, f"Unexpected path length: {len(maptics.path)}"
    assert maptics.path[0] == (0, 0, 0, 15), "Unexpected path coordinates"

def test_prep_tools():
    """Test preparation tools for welding setup."""
    angle_grinder(30, 20000, "water")
    swarf_vacuum()
    acetylene_mark(low_oxy=True, duration=0.8)
    auto_markup(30, 45, 2.0)
    # Note: Hardware-dependent, assuming no errors raised

def test_test_tools():
    """Test tools for weld integrity."""
    assert flex_until_break("5mm/min", "hydraulic") is False, "Flex test should not break"
    ink_test("red_dye", True)
    # Note: Hardware-dependent, assuming no errors raised

def test_post_process():
    """Test post-processing functions for welds."""
    anodize("sulfuric", 20, 20, "pearl_gold", True)
    viscosity_check(20)
    pack("urea", 850, 4)
    quench("mineral_oil", 200)
    paint("epoxy_primer", "polyurethene", "mica_gold")
    # Note: Hardware-dependent, assuming no errors raised

# Example usage
if __name__ == "__main__":
    pytest.main(["-v"])
