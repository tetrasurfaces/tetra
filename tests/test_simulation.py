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

import os
import numpy as np
from tetra.utils.periodic_table import carbon, iron, oganesson
from tetra.utils.rig import Rig
from tetra.tetra.gyrogimbal import TetraVibe, Sym
import pytest

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

def test_gyro_gimbal(tmp_path):
    """Test gyroscopic gimbal with element properties."""
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

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
