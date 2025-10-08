# tests/test_simulation.py
# Copyright 2025 Beau Ayres
# Licensed under Apache-2.0 OR AGPL-3.0-or-later

import pytest
import os
from forge_telemetry import log, flag, rust_probe, depth_error, crack_location
from solid import mesh
from haptics import buzz, shake
from welding import weave, TIG, acetylene

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

def test_forge_telemetry_flag():
    """Test flagging an issue."""
    flag("hydrogen")
    # Placeholder: Add checks if flag logs to CSV

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
