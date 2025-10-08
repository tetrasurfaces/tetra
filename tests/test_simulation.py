# tests/test_simulation.py
# Copyright 2025 Beau Ayres
# Licensed under Apache-2.0 OR AGPL-3.0-or-later

import pytest
import os
from ..forge_telemetry import log, flag, rust_probe, depth_error, crack_location
from ..solid import mesh
from ..haptics import buzz, shake
from ..welding import weave, TIG, acetylene
from ..rig import Rig
from ..friction import Friction
from ..maptics import Maptics
from ..prep_tools import angle_grinder, swarf_vacuum, acetylene_mark, auto_markup
from ..test_tools import flex_until_break, ink_test
from ..post_process import anodize, viscosity_check, pack, quench, paint

def test_forge_telemetry_log(tmp_path):
    """Test logging to CSV."""
    log_file = tmp_path / "weld_log.csv"
    os.environ["TELEMETRY_LOG_FILE"] = str(log_file)  # Override log file path
    log("test_event", amps=60, volts=182)
    assert os.path.exists(log_file)
    with open(log_file, 'r') as f:
        content = f.read()
        assert "test_event" in content
        assert "amps" in content

def test_forge_telemetry_flag(tmp_path):
    """Test flagging an issue."""
    log_file = tmp_path / "weld_log.csv"
    os.environ["TELEMETRY_LOG_FILE"] = str(log_file)
    flag("hydrogen")
    assert os.path.exists(log_file)
    with open(log_file, 'r') as f:
        content = f.read()
        assert "flag_hydrogen" in content

def test_forge_telemetry_probes():
    """Test probe functions."""
    assert rust_probe() == 0
    assert depth_error() is False
    assert crack_location() is None

def test_solid_mesh():
    """Test mesh generation."""
    mesh_data = mesh("W21x62")
    assert mesh_data["type"] == "W21x62"
    assert mesh_data["geometry"] == "hyperbolic_ellipse"

def test_haptics():
    """Test haptic feedback."""
    buzz("low")
    shake("hard")
    # Placeholder: Add checks for haptic output if integrated with hardware

def test_welding():
    """Test welding functions."""
    weave("christmas_tree", speed=18, arc=1.8)
    TIG(tungsten="1%lan", argon=98, volts=25, hz=100)
    acetylene(low_oxy=True, duration=0.8)
    # Placeholder: Add checks for welding parameters

def test_rig():
    """Test rig stabilization."""
    rig = Rig()
    rig.tilt("left", 20)
    rig.stabilize()
    # Placeholder: Add checks for angle or torque

def test_friction():
    """Test friction damping."""
    friction = Friction()
    friction.damp(0.5)
    friction.oscillation()
    assert friction.damping == 0.5

def test_maptics():
    """Test path recording."""
    maptics = Maptics()
    maptics.record_path(0, 0, 0, 15)
    maptics.replay_path()
    assert len(maptics.path) == 1
    assert maptics.path[0] == (0, 0, 0, 15)

def test_prep_tools():
    """Test preparation tools."""
    angle_grinder(30, 20000, "water")
    swarf_vacuum()
    acetylene_mark(low_oxy=True, duration=0.8)
    auto_markup(30, 45, 2.0)
    # Placeholder: Add checks for output

def test_test_tools():
    """Test testing tools."""
    assert flex_until_break("5mm/min", "hydraulic") is False
    ink_test("red_dye", True)
    # Placeholder: Add checks for output

def test_post_process():
    """Test post-processing tools."""
    anodize("sulfuric", 20, 20, "pearl_gold", True)
    viscosity_check(20)
    pack("urea", 850, 4)
    quench("mineral_oil", 200)
    paint("epoxy_primer", "polyurethene", "mica_gold")
    # Placeholder: Add checks for output
