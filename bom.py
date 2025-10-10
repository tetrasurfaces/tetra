# bom.py (Bill-Of-Materials)
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

import sys
import os

# Fallback to add tetra directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from periodic_table import carbon, iron, oganesson

class BOM:
    """Generalized Bill of Materials stub for manufacturer specifications with periodic table support."""
    def __init__(self, component_type, specs=None):
        """
        Initialize BOM with component type and specifications.
        
        Args:
            component_type (str): Type of component (e.g., "electrode", "crane").
            specs (dict, optional): Manufacturer specifications. Defaults to empty dict.
        """
        self.component_type = component_type
        self.specs = specs if specs is not None else {}
        self.elements = {"carbon": carbon, "iron": iron, "oganesson": oganesson}  # Known elements

    def add_spec(self, key, value):
        """Add or update a specification, resolving elements if applicable."""
        if key and value is not None:
            if key == "material" and value.lower() in self.elements:
                element = self.elements[value.lower()]
                self.specs[key] = value.lower()
                self.specs.update({
                    "atomic_weight": getattr(element, "atomic_weight", None),
                    "melting_point": getattr(element, "melting_point", None),
                    "atomic_number": getattr(element, "atomic_number", None)
                })
                print(f"Added {key}: {value} with element properties to {self.component_type} specs")
            else:
                self.specs[key] = value
                print(f"Added {key}: {value} to {self.component_type} specs")
        else:
            print("Invalid key or value provided")

    def get_spec(self, key):
        """Retrieve a specification by key."""
        return self.specs.get(key, None)

    def validate_specs(self):
        """Basic validation of common specification fields."""
        validated = {}
        for key, value in self.specs.items():
            if isinstance(value, (int, float, str)) and value != "":
                validated[key] = value
            else:
                print(f"Warning: Invalid value for {key}: {value}")
        self.specs = validated
        return self.specs

    def resolve_elements(self):
        """Resolve and update specs with element properties and safety ratings."""
        if "material" in self.specs and self.specs["material"].lower() in self.elements:
            element = self.elements[self.specs["material"].lower()]
            self.specs.update({
                "atomic_weight": getattr(element, "atomic_weight", None),
                "melting_point": getattr(element, "melting_point", None),
                "atomic_number": getattr(element, "atomic_number", None),
                "safety_rating_fire": element.safety_rating("fire"),
                "safety_rating_spill": element.safety_rating("spill")
            })
            print(f"Resolved element properties and safety ratings for {self.specs['material']}")
        return self.specs

    def __str__(self):
        """String representation of the BOM."""
        return f"BOM for {self.component_type}: {self.specs}"

if __name__ == "__main__":
    # Example usage
    electrode_bom = BOM("electrode")
    electrode_bom.add_spec("voltage", 180)
    electrode_bom.add_spec("amperage", 50)
    electrode_bom.add_spec("material", "iron")
    electrode_bom.add_spec("invalid_key", None)  # Should warn
    print(electrode_bom)
    validated_specs = electrode_bom.validate_specs()
    print(f"Validated specs: {validated_specs}")
    resolved_specs = electrode_bom.resolve_elements()
    print(f"Resolved specs with elements: {resolved_specs}")
