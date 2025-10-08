# Kappasha: Tetrasurfaces Welding Simulation
## Overview
The tetrasurfaces project, part of the kappasha repository, is a Python-based welding simulation framework designed to model the preparation, welding, and testing of structural beams, with advanced features for porosity modeling, supply chain vector tracking, and quantum-inspired synchronization. It builds on the Tetra Forge prototype, extending it with rhombohedral voxel meshing, IPFS-based navigation, and enhanced telemetry for case hardening, hydrogen cracking prevention, and fleet logistics. The project is dual-licensed under Apache-2.0 and AGPL-3.0-or-later, with Beau Ayres as the copyright holder.

## Features

Mesh Generation: Creates tetrahedral and rhombohedral voxel meshes for beams (e.g., W21x62) using solid.py, tetra.py, and rhombus_voxel.py.
Porosity Modeling: Simulates void formation and martensite layers with porosity_hashing.py, fractal_tetra.py, and rhombohedral voxels.
Welding Simulation: Supports stick, TIG, and acetylene welding (welding.py), with electrode behavior (electrode.py), backstep sequences, and crane sway effects (crane_sway.py).
Telemetry Logging: Logs welding parameters, quench profiles, IPFS navigation, and voxel metrics via rig.py (replaces telemetry.py).
Supply Chain Vectors: Tracks material flow from forge to weld with particle_vector.py and fleet logistics with fleet_vector.py.
IPFS Navigation: Implements decentralized route caching for fleet vectors in rig.py.
Quantum-Inspired Synchronization: Ensures rig telemetry consistency with quantum_sync.py.
Surface Preparation: Handles grinding, marking, and markup with prep_tools.py.
Haptic Feedback: Provides buzz and shake feedback for operator guidance using haptics.py.
Path Recording: Records and replays welder hand paths with maptics.py.
Stabilization: Manages gyroscopic stabilization for torch and jib movements with rig.py and gyrogimbal.py.
Vibration Modeling: Simulates damping and oscillations with frictionvibe.py and friction.py.
Post-Processing: Includes case hardening, anodizing, quenching, and painting via post_process.py.
Testing: Performs flex and dye penetration tests with test_tools.py, validated by test_simulation.py.

## Directory Structure
/home/yeetbow/kappasha/
├── tetrasurfaces/
│   ├── __init__.py
│   ├── tetra.py
│   ├── kappa_grid.py
│   ├── fractal_tetra.py
│   ├── ribit.py
│   ├── ribitstructure.py
│   ├── gyrogimbal.py
│   ├── frictionvibe.py
│   ├── rig.py
│   ├── porosity_hashing.py
│   ├── electrode.py
│   ├── quantum_sync.py
│   ├── fleet_vector.py
│   ├── crane_sway.py
│   ├── particle_vector.py
│   ├── rhombus_voxel.py
│   ├── tetra/
│   │   ├── __init__.py
│   │   ├── forge_telemetry.py
│   │   ├── solid.py
│   │   ├── haptics.py
│   │   ├── welding.py
│   │   ├── rig.py
│   │   ├── friction.py
│   │   ├── maptics.py
│   │   ├── prep_tools.py
│   │   ├── test_tools.py
│   │   └── post_process.py
│   └── tests/
│       ├── __init__.py
│       └── test_simulation.py
├── README.md
└── LICENSE

## Installation

Clone the repository:git clone https://github.com/tetrasurfaces/kappasha.git
cd kappasha/tetrasurfaces


Install dependencies:pip install numpy pytest

If a requirements.txt is provided, use:pip install -r requirements.txt


Set the Python path:export PYTHONPATH=$PYTHONPATH:/home/user/kappasha/tetrasurfaces



## Usage
Run individual modules or the test suite:
# Run rig simulation with telemetry
python rig.py

# Run test suite
pytest tests/test_simulation.py -v

## Example: Simulate a welding sequence with rhombohedral voxel porosity and crane sway:
from rig import Rig
from crane_sway import simulate_crane_sway
from rhombus_voxel import generate_rhombus_voxel

rig = Rig(log_file="weld_log.csv")
rig.tilt("left", 20)
rig.stabilize()
displacements = simulate_crane_sway(beam_length=384, steps=5)
voxel_grid, voids = generate_rhombus_voxel(grid_size=10)
rig.log_voxel_metrics(voxel_grid, len(voids))
rig.log_quench([900, 700, 500, 300, 100, 20])
rig.log_ipfs_navigation([(0, 10, 100), (1, 12, 95)], cache_moves=2)

To customize, modify module parameters (e.g., welding style, material, or rhombus angle) or call specific functions (e.g., electrode.simulate_electrode, quantum_sync.quantum_sync).
Testing
The test suite (test_simulation.py) validates:

Mesh generation (kappa_grid, solid, rhombus_voxel)
Porosity hashing (porosity_hashing)
Electrode behavior (electrode)
Crane sway (crane_sway)
Particle and fleet vectors (particle_vector, fleet_vector)
Quantum synchronization (quantum_sync)
Welding, telemetry, and rig control (rig, welding, forge_telemetry)
Surface preparation, haptics, and post-processing (prep_tools, haptics, post_process)

## Run tests:
cd tetrasurfaces
pytest tests/test_simulation.py -v

## Recent Changes

## October 2025:
Replaced telemetry.py with rig.py, combining rig control (tilt, stabilize) with telemetry logging for welding, quenching, IPFS navigation, and voxel metrics.
Added new modules: porosity.py (void tracking), electrode.py (arc welding), quantum_sync.py (rig synchronization), fleet_vector.py (caster logistics), crane_sway.py (sway simulation), particle_vector.py (supply chain vectors), rhombus_voxel.py (rhombohedral voxel meshing).
Updated fractal_tetra.py to support rhombohedral voxels for porosity simulations.
Enhanced rig.py to log voxel-based metrics (e.g., void density).
Updated test_simulation.py with tests for new modules, including rhombus voxels, and improved error reporting.
Fixed ModuleNotFoundError: kappa_grid in tetra.py by ensuring correct imports.
Enhanced ribs_tructure.py for porosity stiffening in case-hardened steel.
Integrated Tetra Forge features, including surface preparation, haptic feedback, and post-processing.



## License
Copyright 2025 Beau Ayres. Dual-licensed under:

Apache License, Version 2.0: Permits proprietary use without requiring derivative works to be open-sourced.
GNU Affero General Public License v3.0 or later: Requires derivative works to be open-sourced if used over a network.

Proprietary extensions are reserved, and unauthorized copying, distribution, or modification is prohibited without express written permission from Beau Ayres. See the LICENSE file for details.
## Contributing
Contributions are welcome under the dual license terms. To contribute:

## Fork the repository.
Create a branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

For bugs, features, or patent-related extensions, open an issue on GitHub.
## Contact
For inquiries, contact Beau Ayres (details TBD) or open an issue at https://github.com/tetrasurfaces/kappasha.
