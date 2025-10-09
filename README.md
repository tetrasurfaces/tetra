# Kappasha: Tetrasurfaces Welding Simulation
## Overview
The tetrasurfaces project, part of the kappasha repository, is a Python-based welding simulation framework designed to model the preparation, welding, and testing of structural beams, with advanced features for porosity modeling, supply chain vector tracking, quantum-inspired synchronization, and paint mixing. It builds on the Tetra Forge prototype, extending it with rhombohedral voxel meshing, centrifugal force modeling, and enhanced telemetry for case hardening, hydrogen cracking prevention, fleet logistics, and paint emulsification. The project is dual-licensed under Apache-2.0 and AGPL-3.0-or-later, with Beau Ayres as the copyright holder.
## Features

Mesh Generation: Creates tetrahedral and rhombohedral voxel meshes for beams (e.g., W21x62) using solid.py, tetra.py, and rhombus_voxel.py.
Porosity Modeling: Simulates void formation and martensite layers with porosity.py, fractal_tetra.py, and rhombohedral voxels.
Welding Simulation: Supports stick, TIG, and acetylene welding (welding.py), with electrode behavior (electrode.py), backstep sequences, and crane sway effects (crane_sway.py).
Telemetry Logging: Logs welding parameters, quench profiles, IPFS navigation, and voxel metrics via rig.py (replaces telemetry.py).
Supply Chain Vectors: Tracks material flow from forge to weld with particles.py and fleet logistics with fleet_vector.py.
IPFS Navigation: Implements decentralized route caching for fleet vectors in rig.py.
Quantum-Inspired Synchronization: Ensures rig telemetry consistency with sync.py.
Surface Preparation: Handles grinding, marking, and markup with prep_tools.py.
Haptic Feedback: Provides buzz and shake feedback for operator guidance using haptics.py.
Path Recording: Records and replays welder hand paths with maptics.py.
Stabilization: Manages gyroscopic stabilization for torch and jib movements with rig.py and gyro_gimbal.py.
Vibration Modeling: Simulates damping and oscillations with friction_vibe.py and friction.py.
Post-Processing: Includes case hardening, anodizing, quenching, and painting via post_process.py.
Gravity Modeling: Simulates downward forces on particles or beams with gravity.py.
Coriolis Effects: Models deflection in rotating systems for long rolls with coriolis.py.
Centrifugal Modeling: Simulates rotational molding (rotomolding.py) and centrifuge emulsification for paint mixing (centrifuge.py).
Two-Pack Paint Systems: Models mixing and solvent evaporation for epoxy or polyurethane paints with solvents.py.
Optical Effects: Models refractive index changes for non-line-of-sight welding with swing_fog.py and seeing_layer.py.

## Installation

Clone the repository:git clone https://github.com/tetrasurfaces/tetra.git
cd tetra


Install dependencies:pip install numpy pytest

If a requirements.txt is provided, use:pip install -r requirements.txt


Set the Python path:export PYTHONPATH=$PYTHONPATH:/home/user/tetrasurfaces/tetra



## Usage
Run individual modules or the test suite:
# Run centrifuge emulsification
python centrifuge.py

## Run rotomolding simulation
python rotomolding.py

## Run two-pack paint simulation
python solvents.py

## Run test suite
pytest tests/test_simulation.py -v

Example: Simulate paint mixing with centrifuge emulsification and rotomolding:
from rig import Rig
from centrifuge import simulate_centrifuge_emulsification
from rotomolding import simulate_rotomolding
from solvents import simulate_two_pack_paint

rig = Rig(log_file="weld_log.csv")
rig.tilt("left", 20)
rig.stabilize()
distances = simulate_centrifuge_emulsification(droplet_radius=1e-6, steps=5)
forces = simulate_rotomolding(mold_radius=0.5, steps=5)
viscosity, solvent = simulate_two_pack_paint(steps=5)
rig.log("Paint mixing simulation", emulsion_distance=distances[-1], rotomolding_force=forces[-1], viscosity=viscosity[-1], solvent=solvent)

To customize, modify module parameters (e.g., RPM, solvent fraction, or mold radius) or call specific functions (e.g., centrifuge.simulate_centrifuge_emulsification, solvents.simulate_two_pack_paint).
## Testing
The test suite (test_simulation.py) validates:

Mesh generation (kappa_grid, solid, rhombus_voxel)
Porosity hashing (porosity_hashing)
Electrode behavior (electrode)
Crane sway (crane_sway)
Particle and fleet vectors (particle_vector, fleet_vector)
Quantum synchronization (quantum_sync)
Welding, telemetry, and rig control (rig, welding, forge_telemetry)
Surface preparation, haptics, and post-processing (prep_tools, haptics, post_process)
Optical effects (swing_fog, seeing_layer)
Gravity and Coriolis effects (gravity, coriolis)
Centrifugal and paint mixing simulations (centrifuge, rotomolding, solvents)

Run tests:
cd tetrasurfaces
pytest tests/test_simulation.py -v

## Recent Changes

## October 2025:
Replaced telemetry.py with rig.py, combining rig control (tilt, stabilize) with telemetry logging for welding, quenching, IPFS navigation, and voxel metrics.
Added new modules: porosity.py (void tracking), electrode.py (arc welding), sync.py (rig synchronization), fleet_vector.py (caster logistics), crane_sway.py (sway simulation), particles.py (supply chain vectors), rhombus_voxel.py (rhombohedral voxel meshing), swing_fog.py (mist refraction), seeing_layer.py (atmospheric blur), gravity.py (downward forces), coriolis.py (rotational deflections), centrifuge.py (emulsification), rotomolding.py (rotational molding), solvents.py (two-pack paint mixing).
Updated fractal_tetra.py to support rhombohedral voxels for porosity simulations.
Enhanced rig.py to log voxel-based metrics, mirage corrections, and centrifugal effects.
Updated test_simulation.py with tests for new modules and improved error reporting.
Fixed ModuleNotFoundError: kappa_grid in tetra.py by ensuring correct imports.
Enhanced ribstructure.py for porosity stiffening in case-hardened steel.
Integrated Tetra Forge features, including surface preparation, haptic feedback, and post-processing.



## License
Copyright 2025 Beau Ayres. Dual-licensed under:

Apache License, Version 2.0: Permits proprietary use without requiring derivative works to be open-sourced.
GNU Affero General Public License v3.0 or later: Requires derivative works to be open-sourced if used over a network.

Proprietary extensions are reserved, and unauthorized copying, distribution, or modification is prohibited without express written permission from Beau Ayres. See the LICENSE file for details.
## Contributing
Contributions are welcome under the dual license terms. To contribute:

Fork the repository.
Create a branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

For bugs, features, or patent-related extensions, open an issue on GitHub.
## Contact
For inquiries, contact Beau Ayres (details TBD) or open an issue at https://github.com/tetrasurfaces/kappasha.
