# Tetra Forge
Tetra Forge is a Python-based welding simulation prototype designed to model the process of preparing, welding, and testing structural beams. It simulates key aspects of welding, including mesh generation, surface preparation, welding techniques, post-processing, and quality testing, with support for haptic feedback and telemetry logging.
Features

Mesh Generation: Creates tetrahedral meshes for beams (e.g., W21x62) using solid.py.
Telemetry Logging: Logs welding parameters and environmental conditions via forge_telemetry.py.
Surface Preparation: Handles grinding, marking, and markup with prep_tools.py.
Welding Simulation: Supports stick and TIG welding with welding.py.
Haptic Feedback: Provides haptic buzz and shake feedback for errors or guidance using haptics.py.
Path Recording: Records and replays welder hand paths with maptics.py.
Stabilization: Manages gyroscopic stabilization for torch movements with rig.py.
Vibration Modeling: Simulates damping and oscillations with friction.py.
Post-Processing: Includes case hardening, anodizing, and painting via post_process.py.
Testing: Performs flex and dye penetration tests with test_tools.py.

Installation

Clone the repository:git clone https://github.com/tetrasurfaces/tetra.git
cd tetra


Install dependencies (if any, e.g., numpy, scipy):pip install -r requirements.txt


Run the simulation:python tetra_forge.py



Usage
The main script (tetra_forge.py) runs the simulation with default parameters:

Environment: garage
Material: mild_steel
Welding style: stick
Post-processing: case_harden

To customize, modify the main function or call individual functions (beam, prep, weld, post, test) with desired parameters.
License
Dual-licensed under Apache-2.0 and AGPL-3.0-or-later. See individual files for details. Proprietary extensions are reserved, and unauthorized use is prohibited without permission from Beau Ayres.
Contributing
Contributions are welcome! Please submit pull requests or open issues for bugs, features, or patent-related extensions.
Contact
For inquiries, contact Beau Ayres (details TBD).
