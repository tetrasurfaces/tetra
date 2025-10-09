# periodic_table.py
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

from rig import Rig

class Element:
    """Base class for periodic table elements."""
    def __init__(self, symbol, name, atomic_weight, group, period, melting_point=None, density=None):
        self.symbol = symbol
        self.name = name
        self.atomic_weight = atomic_weight  # in u (atomic mass units)
        self.group = group  # IUPAC group number
        self.period = period  # period number
        self.melting_point = melting_point  # in °C, if available
        self.density = density  # in g/cm³, if available
    
    def log_usage(self, context):
        """Log element usage in a simulation context."""
        rig = Rig()
        rig.log(f"Element usage: {self.name}", symbol=self.symbol, atomic_weight=self.atomic_weight, context=context)

# Periodic table data (initial subset)
periodic_table = {
    "hydrogen": Element("H", "Hydrogen", 1.008, 1, 1, melting_point=-259.16, density=0.0899),
    "carbon": Element("C", "Carbon", 12.011, 14, 2, melting_point=3550, density=2.267),
    "nitrogen": Element("N", "Nitrogen", 14.007, 15, 2, melting_point=-210.0, density=1.251),
    "oxygen": Element("O", "Oxygen", 15.999, 16, 2, melting_point=-218.4, density=1.429),
    "iron": Element("Fe", "Iron", 55.845, 8, 4, melting_point=1538, density=7.874),
    "chromium": Element("Cr", "Chromium", 51.996, 6, 4, melting_point=1907, density=7.19),
    "nickel": Element("Ni", "Nickel", 58.693, 10, 4, melting_point=1455, density=8.908)
}

# Allow dynamic access to elements
globals().update(periodic_table)

# Example usage
if __name__ == "__main__":
    carbon.log_usage("weld simulation")
    print(f"Carbon: Weight = {carbon.atomic_weight} u, Group = {carbon.group}, Period = {carbon.period}")
    print(f"Iron: Melting Point = {iron.melting_point} °C, Density = {iron.density} g/cm³")
