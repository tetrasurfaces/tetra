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

from tetra.utils.periodic_table import carbon, iron, oganesson
from tetra.utils.rig import Rig
from tetra.tetra.gyrogimbal import TetraVibe, Sym
from tetra import kappa_grid
from porosity import porosity_hashing
from electrode import simulate_electrode
from crane_sway import simulate_crane_sway
from particle_vector import track_particle_vector
from quantum_sync import quantum_sync
from fleet_vector import simulate_fleet_vector
from rhombus_voxel import generate_rhombus_voxel
from swing_fog import model_swing_fog
from seeing_layer import simulate_seeing_layer
from gravity import simulate_gravity
from coriolis import simulate_coriolis
from centrifuge import simulate_centrifuge_emulsification
from rotomolding import simulate_rotomolding
from solvents import simulate_two_pack_paint
from tetra.solid import mesh
from tetra.haptics import buzz, shake
from tetra.welding import weave, TIG, acetylene
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
    grid = np.random.rand(10, 10, 10)
    hashed_voids = porosity_hashing(grid, void_threshold=0.3)
    assert isinstance(hashed_voids, dict), "Hashed voids should be a dictionary"
    assert len(hashed_voids) > 0, "No voids detected"

def test_rhombus_voxel(tmp_path):
    """Test rhombohedral voxel grid generation and logging."""
    voxel_grid, voids = generate_rhombus_voxel(grid_size=10)
    assert voxel_grid.shape == (10, 10, 10), f"Unexpected voxel grid shape: {voxel_grid.shape}"
    assert isinstance(voids, dict), "Voids should be a dictionary"
    rig = Rig(log_file=str(tmp_path / "weld_log.csv"))
    rig.log_voxel_metrics(voxel_grid, len(voids))
    assert os.path.exists(rig.log_file), "Log file not created"
    with open(rig.log_file, 'r') as f:
        content = f.read()
        assert "Voxel analysis" in content, "Voxel metrics not logged"

def test_electrode_stability():
    """Test electrode simulation for arc stability."""
    result = simulate_electrode(voltage=180, amperage=50, arc_length=3, electrode_gap=2)
    assert result['arc_stability'] > 0.7, f"Arc stability too low: {result['arc_stability']}"
    assert result['hydrogen_content'] < 4, f"Hydrogen content too high: {result['hydrogen_content']}"

def test_crane_sway():
    """Test crane sway simulation."""
    displacements = simulate_crane_sway(beam_length=384, steps=5)
    assert len(displacements) == 5, f"Unexpected number of sway displacements: {len(displacements)}"
    assert all(abs(d) < 10 for d in displacements), "Sway displacement too large"

def test_particle_vector():
    """Test particle vector tracking."""
    stages = [(0, 0, 0, 'forge'), (10, 5, 0, 'ship'), (15, 5, 2, 'weld')]
    vectors = track_particle_vector(stages)
    assert len(vectors) == 3, f"Unexpected number of particle vectors: {len(vectors)}"
    assert all(len(v) == 3 for v in vectors), "Invalid vector dimensions"

def test_quantum_sync():
    """Test quantum-inspired synchronization."""
    rig1 = {'temp': 850, 'crown': 0.1}
    rig2 = {'temp': 849, 'crown': 0.12}
    sync_status = quantum_sync(rig1, rig2, tolerance=0.1)
    assert sync_status, "Rigs should be synchronized within tolerance"

def test_fleet_vector():
    """Test fleet vector simulation."""
    casters = [(1, 0, 10, 100), (2, 1, 12, 95), (3, 2, 11, 98)]
    meta_vec, hashes = simulate_fleet_vector(casters)
    assert isinstance(meta_vec, dict), "Meta-vector should be a dictionary"
    assert len(hashes) <= 5, f"Too many IPFS hashes: {len(hashes)}"

def test_forge_telemetry_log(tmp_path):
    """Test logging to CSV."""
    log_file = tmp_path / "weld_log.csv"
    rig = Rig(log_file=str(log_file))
    rig.log("test_event", amps=60, volts=182)
    assert os.path.exists(log_file), "Log file not created"
    with open(log_file, 'r') as f:
        content = f.read()
        assert "test_event" in content, "Event not logged"

def test_forge_telemetry_flag(tmp_path):
    """Test flagging an issue."""
    log_file = tmp_path / "weld_log.csv"
    rig = Rig(log_file=str(log_file))
    rig.flag("hydrogen")
    assert os.path.exists(log_file), "Log file not created"
    with open(log_file, 'r') as f:
        content = f.read()
        assert "flag_hydrogen" in content, "Hydrogen flag not logged"

def test_forge_telemetry_probes():
    """Test probe functions."""
    rig = Rig()
    assert rig.rust_probe() == 0, "Rust probe should return 0"
    assert rig.depth_error() is False, "Depth error should be False"
    assert rig.crack_location() is None, "Crack location should be None"

def test_swing_fog():
    """Test swing fog refraction modeling."""
    bend = model_swing_fog(distance=40, index=1.002)
    assert bend > 0, "Bend radius should be positive"

def test_seeing_layer():
    """Test seeing layer blur simulation."""
    blur = simulate_seeing_layer(heat_delta=0.5, angle=45)
    assert blur > 0, "Blur radius should be positive"

def test_rig_mirage(tmp_path):
    """Test mirage correction logging."""
    log_file = tmp_path / "weld_log.csv"
    rig = Rig(log_file=str(log_file))
    rig.log_mirage(heat_temp=900, air_temp=30)
    assert os.path.exists(log_file), "Log file not created"
    with open(log_file, 'r') as f:
        content = f.read()
        assert "Mirage correction" in content, "Mirage log not created"

def test_gravity():
    """Test gravity simulation."""
    displ = simulate_gravity(steps=5)
    assert len(displ) == 5, "Unexpected number of gravity displacements"
    assert all(d > 0 for d in displ), "Displacements should be positive"

def test_coriolis():
    """Test Coriolis simulation."""
    forces = simulate_coriolis(steps=5)
    assert len(forces) == 5, "Unexpected number of Coriolis forces"
    assert all(isinstance(f, np.ndarray) for f in forces), "Forces should be numpy arrays"

def test_centrifuge_emulsification():
    """Test centrifuge emulsification with Coriolis simulation."""
    displacements = simulate_centrifuge_emulsification(steps=5, latitude=35.0)
    assert len(displacements) == 5, "Unexpected number of centrifuge displacements"
    assert all(d >= 0 for d in displacements), "Displacements should be non-negative"
    assert any(abs(d) > 0 for d in displacements), "Some displacement should occur"

def test_rotomolding():
    """Test rotomolding simulation."""
    forces = simulate_rotomolding(steps=5)
    assert len(forces) == 5, "Unexpected number of rotomolding forces"
    assert all(f >= 0 for f in forces), "Forces should be non-negative"

def test_two_pack_paint(tmp_path):
    """Test two-pack paint simulation."""
    log_file = tmp_path / "weld_log.csv"
    viscosity, solvent = simulate_two_pack_paint(steps=5)
    assert len(viscosity) == 5, "Unexpected viscosity profile length"
    assert solvent >= 0, "Solvent fraction should be non-negative"
    assert os.path.exists(log_file), "Log file not created"
    with open(log_file, 'r') as f:
        content = f.read()
        assert "Two-pack paint mixing" in content, "Paint mixing not logged"

def test_solid_mesh():
    """Test mesh generation."""
    mesh_data = mesh("W21x62")
    assert mesh_data["type"] == "W21x62", f"Unexpected mesh type: {mesh_data['type']}"
    assert mesh_data["geometry"] == "hyperbolic_ellipse", f"Unexpected geometry: {mesh_data['geometry']}"

def test_haptics():
    """Test haptic feedback."""
    buzz("low")
    shake("hard")
    # Note: Hardware-dependent, assuming no errors raised

def test_welding():
    """Test welding functions."""
    weave("christmas_tree", speed=18, arc=1.8)
    TIG(tungsten="1%lan", argon=98, volts=25, hz=100)
    acetylene(low_oxy=True, duration=0.8)
    # Note: Hardware-dependent, assuming no errors raised

def test_friction():
    """Test friction damping."""
    friction = Friction()
    friction.damp(0.5)
    friction.oscillation()
    assert friction.damping == 0.5, f"Unexpected damping value: {friction.damping}"

def test_maptics():
    """Test path recording."""
    maptics = Maptics()
    maptics.record_path(0, 0, 0, 15)
    maptics.replay_path()
    assert len(maptics.path) == 1, f"Unexpected path length: {len(maptics.path)}"
    assert maptics.path[0] == (0, 0, 0, 15), "Unexpected path coordinates"

def test_prep_tools():
    """Test preparation tools."""
    angle_grinder(30, 20000, "water")
    swarf_vacuum()
    acetylene_mark(low_oxy=True, duration=0.8)
    auto_markup(30, 45, 2.0)
    # Note: Hardware-dependent, assuming no errors raised

def test_post_process():
    """Test post-processing."""
    anodize("sulfuric", 20, 20, "pearl_gold", True)
    viscosity_check(20)
    pack("urea", 850, 4)
    quench("mineral_oil", 200)
    paint("epoxy_primer", "polyurethene", "mica_gold")
    # Note: Hardware-dependent, assuming no errors raised"

def test_periodic_table_extended(tmp_path):
    """Test expanded periodic table and structure mapping."""
    log_file = tmp_path / "weld_log.csv"
    rig = Rig(log_file=str(log_file))
    carbon.log_usage("test simulation")
    iron.log_usage("test simulation")
    oganesson.log_usage("test simulation")
    assert carbon.atomic_weight == 12.011, f"Unexpected carbon weight: {carbon.atomic_weight}"
    assert iron.melting_point == 1538, f"Unexpected iron melting point: {iron.melting_point}"
    assert oganesson.atomic_number == 118, f"Unexpected oganesson number: {oganesson.atomic_number}"
    carbon_map = carbon.map_structure(spin_rate=1.0, friction_coeff=0.1)
    assert "gravitational_force" in carbon_map, "Structure map missing gravitational force"
    assert os.path.exists(log_file), "Log file not created"
    with open(log_file, 'r') as f:
        content = f.read()
        assert "Element usage: Carbon" in content, "Carbon usage not logged"
        assert "Element usage: Iron" in content, "Iron usage not logged"
        assert "Element usage: Oganesson" in content, "Oganesson usage not logged"

def test_gyro_gimbal_and_sym(tmp_path):
    """Test gyroscopic gimbal and Sym class functionality."""
    log_file = tmp_path / "weld_log.csv"
    rig = Rig(log_file=str(log_file))
    model = TetraVibe()
    sym = Sym()
    pos1 = np.array([0, 0, 0])
    pos2 = np.array([0.05, 0, 0])
    wave, spin, force = model.gyro_gimbal(pos1, pos2, element=carbon, spin_rate=1.0)
    assert wave > 0, "Wave should be positive"
    assert len(spin) == 3, "Spin should be 3D vector"
    assert force > 0, "Force should be positive"
    sym.tilt("spin_axis", 1.0)
    sym.stabilize()
    assert sym.spin_rate < 1.0, "Spin rate should be damped"
    spin_vector = sym.get_spin_vector()
    assert len(spin_vector) == 3, "Spin vector should be 3D"
    assert np.allclose(np.linalg.norm(spin_vector), 1.0, atol=1e-6), "Spin vector magnitude should be close to 1.0 after stabilization"
    rig.log_element_space_properties(carbon, spin_rate=1.0, friction_coeff=0.1)
    with open(log_file, 'r') as f:
        content = f.read()
        assert "Space properties for Carbon" in content, "Carbon space properties not logged"
        assert "spin_vector" in content, "Spin vector not logged"

def test_rig_space_properties(tmp_path):
    """Test rig logging of element space properties."""
    log_file = tmp_path / "weld_log.csv"
    rig = Rig(log_file=str(log_file))
    rig.log_element_space_properties(carbon, spin_rate=1.5, friction_coeff=0.15)
    assert os.path.exists(log_file), "Log file not created"
    with open(log_file, 'r') as f:
        content = f.read()
        assert "Space properties for Carbon" in content, "Carbon space properties not logged"
        assert "gravitational_force" in content, "Gravitational force not logged"
        assert "spin_vector" in content, "Spin vector not logged"
    rig.log_element_space_properties("invalid", spin_rate=1.0, friction_coeff=0.1)
    with open(log_file, 'r') as f:
        content = f.read()
        assert "Invalid element: invalid" in content, "Invalid element logging failed"

# Example usage
if __name__ == "__main__":
    pytest.main(["-v"])
