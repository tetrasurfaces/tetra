# factory_sim.py
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

import simpy
import random
from utils.rig import Rig
from tetra.utils.periodic_table import Element
from bom import BOM
from crane import Crane

class FactorySim:
    """Simulate a welding factory with safety SOPs."""
    def __init__(self, env, num_welders=5, num_beams=10):
        self.env = env
        self.num_welders = num_welders
        self.num_beams = num_beams
        self.welders = simpy.Resource(env, capacity=num_welders)
        self.rig = Rig()
        self.crane = Crane(beam_length=384.0)
        self.bom = BOM("welding_beam")
        self.bom.add_spec("material", "iron")
        self.bom.resolve_elements()
        self.element = self.bom.elements.get(self.bom.specs["material"], Element("Fe", "Iron", 26, 55.845, 8, 4))

    def safety_check(self):
        """Perform safety check based on SOPs."""
        # Example SOPs from search results
        sop_steps = [
            "Ensure welding area is ventilated and free of flammable materials.",
            "Wear appropriate PPE: welding helmet, gloves, apron.",
            "Check equipment for damage and proper grounding.",
            "Verify emergency exits and fire extinguishers are accessible.",
            "Confirm training on element hazards (e.g., hydrogen content)."
        ]
        for step in sop_steps:
            self.rig.log("Safety SOP", step=step)
        # Check element risk
        risk_profile = self.element.generate_risk_profile("fire")
        self.rig.log("Element Risk Profile", profile=risk_profile)
        return True  # Assume passes; in real sim, could randomize failures

    def weld_process(self, beam_id):
        """Simulate welding process for a beam."""
        if not self.safety_check():
            self.rig.flag("Safety failure")
            return
        
        with self.welders.request() as req:
            yield req
            # Simulate preparation
            yield self.env.timeout(random.randint(5, 10))  # Prep time
            self.rig.log("Preparation complete", beam_id=beam_id)
            
            # Crane sway simulation
            self.crane.set_wind_direction(random.uniform(0, 360))
            sway = self.crane.simulate_crane_sway(steps=5)
            self.rig.log("Crane sway", average_sway=np.mean(sway))
            
            # Welding
            yield self.env.timeout(random.randint(10, 20))  # Weld time
            self.rig.log("Welding complete", beam_id=beam_id)
            
            # Post-processing
            yield self.env.timeout(random.randint(5, 15))  # Post time
            self.rig.log("Post-processing complete", beam_id=beam_id)

    def run(self):
        """Run the factory simulation."""
        for beam_id in range(self.num_beams):
            self.env.process(self.weld_process(beam_id))
        self.env.run(until=100)  # Simulate for 100 time units

if __name__ == "__main__":
    env = simpy.Environment()
    factory = FactorySim(env)
    factory.run()
    print("Factory simulation complete")
