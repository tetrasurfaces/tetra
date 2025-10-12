# arch_utils.py
# !/usr/bin/env python3
# Copyright 2025 Beau Ayers, xAI
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

import numpy as np
import json
import os
from kappasha256 import hash_surface

def read_config(config_file="config/config.json"):
    """Read intent and commercial use from config file with error handling."""
    config_dir = os.path.dirname(config_file)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    if not os.path.exists(config_file):
        print(f"Config file {config_file} not found. Creating default.")
        write_config("none", False, config_file)
        return None, False
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        intent = config.get("intent")
        commercial_use = config.get("commercial_use", False)
        if intent not in ["educational", "commercial", "none"]:
            raise ValueError("Invalid intent in config.")
        return intent, commercial_use
    except json.JSONDecodeError:
        print(f"Error: {config_file} contains invalid JSON. Resetting to default.")
        write_config("none", False, config_file)
        return None, False
    except Exception as e:
        print(f"Error reading {config_file}: {e}. Resetting to default.")
        write_config("none", False, config_file)
        return None, False

def write_config(intent, commercial_use, config_file="config/config.json"):
    """Write intent and commercial use to config file with error handling."""
    config = {"intent": intent, "commercial_use": commercial_use}
    config_dir = os.path.dirname(config_file)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error writing to {config_file}: {e}")

def check_license(commercial_use=False, intent=None):
    """Ensure license compliance and intent declaration."""
    if intent not in ["educational", "commercial"]:
        notice = """
        NOTICE: You must declare your intent to use this software.
        - For educational use (e.g., university training), open a GitHub issue at github.com/tetrasurfaces/issues using the Educational License Request template.
        - For commercial use (e.g., branding, molding), use the Commercial License Request template.
        See NOTICE.txt for details. Do not share proprietary details in public issues.
        """
        print(f"License check failed: Invalid or missing intent. {notice}")
        raise ValueError("Invalid or missing intent.")
    if commercial_use and intent != "commercial":
        notice = "Commercial use requires 'commercial' intent and a negotiated license via github.com/tetrasurfaces/issues."
        print(f"License check failed: {notice}")
        raise ValueError(notice)
    return True

def tetra_hash_surface(mesh, precision=6):
    """Hash a VTK mesh using kappasha256 for CAD, site, or render sync."""
    intent, commercial_use = read_config()
    check_license(commercial_use, intent)
    vertices = np.array([mesh.GetPoint(i) for i in range(mesh.GetNumberOfPoints())])
    hash_val = hash_surface(vertices, precision=precision)
    return hash_val

def calc_live_kappa(mesh, target=0.5):
    """Calculate curvature delta from target (positive: too bendy, negative: too flat)."""
    intent, commercial_use = read_config()
    check_license(commercial_use, intent)
    points = np.array([mesh.GetPoint(i) for i in range(mesh.GetNumberOfPoints())])
    kappa = 0.0  # Placeholder - plug real VTK curvature filter
    return kappa - target

def apply_tetra_etch(mesh, depth=0.01, hash_val=None):
    """Embed hash into surface as geometric offset for render or physical etch."""
    intent, commercial_use = read_config()
    check_license(commercial_use, intent)
    if hash_val is None:
        hash_val = tetra_hash_surface(mesh)
    for i in range(mesh.GetNumberOfPoints()):
        pt = np.array(mesh.GetPoint(i))
        bump = np.sin(i * 0.8 + float(hash_val)) * depth * 0.5
        pt[2] += bump  # Z-offset only
        mesh.SetPoint(i, pt)
    return hash_val
