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

import numpy as np
from rig import Rig
from gyrogimbal import Rig as GyroRig  # Import from gyrogimbal for gyroscopic modeling

class Element:
    """Base class for periodic table elements."""
    def __init__(self, symbol, name, atomic_weight, group, period, melting_point=None, boiling_point=None, density=None, atomic_radius=None, electronegativity=None, space_gravity_constant=6.67430e-11):
        self.symbol = symbol
        self.name = name
        self.atomic_weight = atomic_weight  # in u (atomic mass units)
        self.group = group  # IUPAC group number
        self.period = period  # period number
        self.melting_point = melting_point  # in °C, if available
        self.boiling_point = boiling_point  # in °C, if available
        self.density = density  # in g/cm³, if available
        self.atomic_radius = atomic_radius  # in pm (picometers), if available
        self.electronegativity = electronegativity  # Pauling scale, if available
        self.space_gravity_constant = space_gravity_constant  # Gravitational constant G for space simulations (m³ kg⁻¹ s⁻²)
    
    def log_usage(self, context):
        """Log element usage in a simulation context."""
        rig = Rig()
        rig.log(f"Element usage: {self.name}", symbol=self.symbol, atomic_weight=self.atomic_weight, context=context)
    
    def map_structure(self, spin_rate=1.0, friction_coeff=0.1, mass_scale=1e-26):
        """Simulate element structure using gimbal_gyro for spin, friction, and gravitational effects."""
        gyro = GyroRig()
        gyro.tilt("spin_axis", spin_rate)  # Simulate atomic spin
        # Gravitational force between two atoms (simplified)
        force_grav = self.space_gravity_constant * (mass_scale * self.atomic_weight) ** 2 / (self.atomic_radius * 1e-12) ** 2  # Convert pm to m
        # Friction in atomic interactions (fictional)
        friction_force = friction_coeff * force_grav
        gyro.stabilize()  # Stabilize after spin
        return {
            "spin_rate": spin_rate,
            "friction_force": friction_force,
            "grav_force": force_grav
        }

# Periodic table data (expanded with more elements and properties)
periodic_table = {
    "hydrogen": Element("H", "Hydrogen", 1.008, 1, 1, melting_point=-259.16, boiling_point=-252.87, density=0.0899, atomic_radius=53, electronegativity=2.20),
    "helium": Element("He", "Helium", 4.0026, 18, 1, melting_point=-272.2, boiling_point=-268.93, density=0.1785, atomic_radius=31, electronegativity=None),
    "lithium": Element("Li", "Lithium", 6.94, 1, 2, melting_point=180.5, boiling_point=1342, density=0.534, atomic_radius=152, electronegativity=0.98),
    "beryllium": Element("Be", "Beryllium", 9.0122, 2, 2, melting_point=1287, boiling_point=2469, density=1.848, atomic_radius=112, electronegativity=1.57),
    "boron": Element("B", "Boron", 10.81, 13, 2, melting_point=2076, boiling_point=3927, density=2.34, atomic_radius=85, electronegativity=2.04),
    "carbon": Element("C", "Carbon", 12.011, 14, 2, melting_point=3550, boiling_point=4827, density=2.267, atomic_radius=77, electronegativity=2.55),
    "nitrogen": Element("N", "Nitrogen", 14.007, 15, 2, melting_point=-210.0, boiling_point=-195.79, density=1.251, atomic_radius=75, electronegativity=3.04),
    "oxygen": Element("O", "Oxygen", 15.999, 16, 2, melting_point=-218.4, boiling_point=-183.0, density=1.429, atomic_radius=73, electronegativity=3.44),
    "fluorine": Element("F", "Fluorine", 18.998, 17, 2, melting_point=-219.67, boiling_point=-188.11, density=1.696, atomic_radius=71, electronegativity=3.98),
    "neon": Element("Ne", "Neon", 20.1797, 18, 2, melting_point=-248.59, boiling_point=-246.05, density=0.8999, atomic_radius=69, electronegativity=None),
    "sodium": Element("Na", "Sodium", 22.9898, 1, 3, melting_point=97.72, boiling_point=883, density=0.968, atomic_radius=186, electronegativity=0.93),
    "magnesium": Element("Mg", "Magnesium", 24.305, 2, 3, melting_point=650, boiling_point=1090, density=1.738, atomic_radius=160, electronegativity=1.31),
    "aluminum": Element("Al", "Aluminum", 26.9815, 13, 3, melting_point=660.32, boiling_point=2519, density=2.70, atomic_radius=143, electronegativity=1.61),
    "silicon": Element("Si", "Silicon", 28.085, 14, 3, melting_point=1414, boiling_point=3265, density=2.329, atomic_radius=111, electronegativity=1.90),
    "phosphorus": Element("P", "Phosphorus", 30.9738, 15, 3, melting_point=44.15, boiling_point=280.5, density=1.823, atomic_radius=110, electronegativity=2.19),
    "sulfur": Element("S", "Sulfur", 32.06, 16, 3, melting_point=115.21, boiling_point=444.6, density=2.07, atomic_radius=104, electronegativity=2.58),
    "chlorine": Element("Cl", "Chlorine", 35.45, 17, 3, melting_point=-101.5, boiling_point=-34.04, density=3.214, atomic_radius=99, electronegativity=3.16),
    "argon": Element("Ar", "Argon", 39.948, 18, 3, melting_point=-189.34, boiling_point=-185.85, density=1.784, atomic_radius=97, electronegativity=None),
    "potassium": Element("K", "Potassium", 39.0983, 1, 4, melting_point=63.38, boiling_point=759, density=0.862, atomic_radius=227, electronegativity=0.82),
    "calcium": Element("Ca", "Calcium", 40.078, 2, 4, melting_point=842, boiling_point=1484, density=1.55, atomic_radius=197, electronegativity=1.00),
    "scandium": Element("Sc", "Scandium", 44.9559, 3, 4, melting_point=1541, boiling_point=2836, density=2.989, atomic_radius=162, electronegativity=1.36),
    "titanium": Element("Ti", "Titanium", 47.867, 4, 4, melting_point=1668, boiling_point=3287, density=4.506, atomic_radius=147, electronegativity=1.54),
    "vanadium": Element("V", "Vanadium", 50.9415, 5, 4, melting_point=1910, boiling_point=3407, density=6.11, atomic_radius=134, electronegativity=1.63),
    "chromium": Element("Cr", "Chromium", 51.996, 6, 4, melting_point=1907, boiling_point=2671, density=7.19, atomic_radius=128, electronegativity=1.66),
    "manganese": Element("Mn", "Manganese", 54.938, 7, 4, melting_point=1246, boiling_point=2061, density=7.21, atomic_radius=127, electronegativity=1.55),
    "iron": Element("Fe", "Iron", 55.845, 8, 4, melting_point=1538, boiling_point=2862, density=7.874, atomic_radius=126, electronegativity=1.83),
    "cobalt": Element("Co", "Cobalt", 58.933, 9, 4, melting_point=1495, boiling_point=2927, density=8.90, atomic_radius=125, electronegativity=1.88),
    "nickel": Element("Ni", "Nickel", 58.693, 10, 4, melting_point=1455, boiling_point=2913, density=8.908, atomic_radius=124, electronegativity=1.91),
    "copper": Element("Cu", "Copper", 63.546, 11, 4, melting_point=1084.62, boiling_point=2562, density=8.96, atomic_radius=128, electronegativity=1.90),
    "zinc": Element("Zn", "Zinc", 65.38, 12, 4, melting_point=419.53, boiling_point=907, density=7.14, atomic_radius=134, electronegativity=1.65),
    "gallium": Element("Ga", "Gallium", 69.723, 13, 4, melting_point=29.76, boiling_point=2400, density=5.91, atomic_radius=135, electronegativity=1.81),
    "germanium": Element("Ge", "Germanium", 72.63, 14, 4, melting_point=938.25, boiling_point=2833, density=5.323, atomic_radius=122, electronegativity=2.01),
    "arsenic": Element("As", "Arsenic", 74.9216, 15, 4, melting_point=816.8, boiling_point=614, density=5.727, atomic_radius=119, electronegativity=2.18),
    "selenium": Element("Se", "Selenium", 78.971, 16, 4, melting_point=221, boiling_point=685, density=4.809, atomic_radius=117, electronegativity=2.55),
    "bromine": Element("Br", "Bromine", 79.904, 17, 4, melting_point=-7.2, boiling_point=58.8, density=3.12, atomic_radius=114, electronegativity=2.96),
    "krypton": Element("Kr", "Krypton", 83.798, 18, 4, melting_point=-157.37, boiling_point=-153.22, density=3.749, atomic_radius=110, electronegativity=None),
    # ... (Additional elements can be added as needed, up to element 118)
}

# Allow dynamic access to elements
globals().update(periodic_table)

# Example usage
if __name__ == "__main__":
    carbon.log_usage("weld simulation")
    print(f"Carbon: Weight = {carbon.atomic_weight} u, Group = {carbon.group}, Period = {carbon.period}, Boiling Point = {carbon.boiling_point} °C")
    print(f"Iron: Melting Point = {iron.melting_point} °C, Density = {iron.density} g/cm³, Atomic Radius = {iron.atomic_radius} pm, Electronegativity = {iron.electronegativity}")
    # Map structure using gimbal_gyro
    carbon_map = carbon.map_structure(spin_rate=1.0, friction_coeff=0.1)
    print(f"Carbon structure map: {carbon_map}")
