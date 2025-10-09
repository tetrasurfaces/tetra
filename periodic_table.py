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
from tetra.gyrogimbal import Sym as GyroRig  # Adjusted import for gyro_gimbal.py in tetra/

class Element:
    """Base class for periodic table elements with space-related properties."""
    def __init__(self, symbol, name, atomic_number, atomic_weight, group, period, melting_point=None, boiling_point=None, density=None, atomic_radius=None, electronegativity=None, space_gravity_constant=6.67430e-11):
        self.symbol = symbol
        self.name = name
        self.atomic_number = atomic_number  # Atomic number (1-118)
        self.atomic_weight = atomic_weight  # in u (atomic mass units)
        self.group = group  # IUPAC group number
        self.period = period  # period number
        self.melting_point = melting_point  # in °C, if available
        self.boiling_point = boiling_point  # in °C, if available
        self.density = density  # in g/cm³, if available
        self.atomic_radius = atomic_radius  # in pm (picometers), if available
        self.electronegativity = electronegativity  # Pauling scale, if available
        self.space_gravity_constant = space_gravity_constant  # Gravitational constant G (m³ kg⁻¹ s⁻²)
    
    def log_usage(self, context):
        """Log element usage in a simulation context."""
        rig = Rig()
        rig.log(f"Element usage: {self.name}", symbol=self.symbol, atomic_weight=self.atomic_weight, context=context)
    
    def map_structure(self, spin_rate=1.0, friction_coeff=0.1, mass_scale=1e-26):
        """Simulate element structure using gyrogimbal for spin, friction, and gravitational effects."""
        gyro = GyroRig()
        gyro.tilt("spin_axis", spin_rate)  # Simulate atomic spin
        # Convert atomic weight to approximate mass (kg) using mass_scale
        atomic_mass_kg = mass_scale * self.atomic_weight
        # Gravitational force between two atoms (simplified, assuming radius as distance)
        force_grav = self.space_gravity_constant * (atomic_mass_kg ** 2) / ((self.atomic_radius * 1e-12) ** 2) if self.atomic_radius else 0.0
        # Friction force (fictional, proportional to gravitational force)
        friction_force = friction_coeff * force_grav if force_grav else 0.0
        gyro.stabilize()  # Stabilize after spin
        return {
            "atomic_number": self.atomic_number,
            "spin_rate": spin_rate,
            "friction_force": friction_force,
            "gravitational_force": force_grav
        }

# Complete periodic table data (up to element 118)
periodic_table = {
    "hydrogen": Element("H", "Hydrogen", 1, 1.008, 1, 1, -259.16, -252.87, 0.0899, 53, 2.20),
    "helium": Element("He", "Helium", 2, 4.0026, 18, 1, -272.2, -268.93, 0.1785, 31, None),
    "lithium": Element("Li", "Lithium", 3, 6.94, 1, 2, 180.5, 1342, 0.534, 152, 0.98),
    "beryllium": Element("Be", "Beryllium", 4, 9.0122, 2, 2, 1287, 2469, 1.848, 112, 1.57),
    "boron": Element("B", "Boron", 5, 10.81, 13, 2, 2076, 3927, 2.34, 85, 2.04),
    "carbon": Element("C", "Carbon", 6, 12.011, 14, 2, 3550, 4827, 2.267, 77, 2.55),
    "nitrogen": Element("N", "Nitrogen", 7, 14.007, 15, 2, -210.0, -195.79, 1.251, 75, 3.04),
    "oxygen": Element("O", "Oxygen", 8, 15.999, 16, 2, -218.4, -183.0, 1.429, 73, 3.44),
    "fluorine": Element("F", "Fluorine", 9, 18.998, 17, 2, -219.67, -188.11, 1.696, 71, 3.98),
    "neon": Element("Ne", "Neon", 10, 20.1797, 18, 2, -248.59, -246.05, 0.8999, 69, None),
    "sodium": Element("Na", "Sodium", 11, 22.9898, 1, 3, 97.72, 883, 0.968, 186, 0.93),
    "magnesium": Element("Mg", "Magnesium", 12, 24.305, 2, 3, 650, 1090, 1.738, 160, 1.31),
    "aluminum": Element("Al", "Aluminum", 13, 26.9815, 13, 3, 660.32, 2519, 2.70, 143, 1.61),
    "silicon": Element("Si", "Silicon", 14, 28.085, 14, 3, 1414, 3265, 2.329, 111, 1.90),
    "phosphorus": Element("P", "Phosphorus", 15, 30.9738, 15, 3, 44.15, 280.5, 1.823, 110, 2.19),
    "sulfur": Element("S", "Sulfur", 16, 32.06, 16, 3, 115.21, 444.6, 2.07, 104, 2.58),
    "chlorine": Element("Cl", "Chlorine", 17, 35.45, 17, 3, -101.5, -34.04, 3.214, 99, 3.16),
    "argon": Element("Ar", "Argon", 18, 39.948, 18, 3, -189.34, -185.85, 1.784, 97, None),
    "potassium": Element("K", "Potassium", 19, 39.0983, 1, 4, 63.38, 759, 0.862, 227, 0.82),
    "calcium": Element("Ca", "Calcium", 20, 40.078, 2, 4, 842, 1484, 1.55, 197, 1.00),
    "scandium": Element("Sc", "Scandium", 21, 44.9559, 3, 4, 1541, 2836, 2.989, 162, 1.36),
    "titanium": Element("Ti", "Titanium", 22, 47.867, 4, 4, 1668, 3287, 4.506, 147, 1.54),
    "vanadium": Element("V", "Vanadium", 23, 50.9415, 5, 4, 1910, 3407, 6.11, 134, 1.63),
    "chromium": Element("Cr", "Chromium", 24, 51.996, 6, 4, 1907, 2671, 7.19, 128, 1.66),
    "manganese": Element("Mn", "Manganese", 25, 54.938, 7, 4, 1246, 2061, 7.21, 127, 1.55),
    "iron": Element("Fe", "Iron", 26, 55.845, 8, 4, 1538, 2862, 7.874, 126, 1.83),
    "cobalt": Element("Co", "Cobalt", 27, 58.933, 9, 4, 1495, 2927, 8.90, 125, 1.88),
    "nickel": Element("Ni", "Nickel", 28, 58.693, 10, 4, 1455, 2913, 8.908, 124, 1.91),
    "copper": Element("Cu", "Copper", 29, 63.546, 11, 4, 1084.62, 2562, 8.96, 128, 1.90),
    "zinc": Element("Zn", "Zinc", 30, 65.38, 12, 4, 419.53, 907, 7.14, 134, 1.65),
    "gallium": Element("Ga", "Gallium", 31, 69.723, 13, 4, 29.76, 2400, 5.91, 135, 1.81),
    "germanium": Element("Ge", "Germanium", 32, 72.63, 14, 4, 938.25, 2833, 5.323, 122, 2.01),
    "arsenic": Element("As", "Arsenic", 33, 74.9216, 15, 4, 816.8, 614, 5.727, 119, 2.18),
    "selenium": Element("Se", "Selenium", 34, 78.971, 16, 4, 221, 685, 4.809, 117, 2.55),
    "bromine": Element("Br", "Bromine", 35, 79.904, 17, 4, -7.2, 58.8, 3.12, 114, 2.96),
    "krypton": Element("Kr", "Krypton", 36, 83.798, 18, 4, -157.37, -153.22, 3.749, 110, None),
    "rubidium": Element("Rb", "Rubidium", 37, 85.4678, 1, 5, 39.31, 688, 1.532, 248, 0.82),
    "strontium": Element("Sr", "Strontium", 38, 87.62, 2, 5, 777, 1382, 2.64, 215, 0.95),
    "yttrium": Element("Y", "Yttrium", 39, 88.9059, 3, 5, 1522, 3338, 4.472, 180, 1.22),
    "zirconium": Element("Zr", "Zirconium", 40, 91.224, 4, 5, 1855, 4409, 6.52, 160, 1.33),
    "niobium": Element("Nb", "Niobium", 41, 92.9064, 5, 5, 2477, 4744, 8.57, 146, 1.6),
    "molybdenum": Element("Mo", "Molybdenum", 42, 95.95, 6, 5, 2623, 4639, 10.28, 139, 2.16),
    "technetium": Element("Tc", "Technetium", 43, 98.0, 7, 5, 2157, 4265, 11.5, 136, 1.9),
    "ruthenium": Element("Ru", "Ruthenium", 44, 101.07, 8, 5, 2334, 4150, 12.45, 134, 2.2),
    "rhodium": Element("Rh", "Rhodium", 45, 102.9055, 9, 5, 1964, 3695, 12.41, 134, 2.28),
    "palladium": Element("Pd", "Palladium", 46, 106.42, 10, 5, 1554.9, 2963, 12.02, 137, 2.20),
    "silver": Element("Ag", "Silver", 47, 107.8682, 11, 5, 961.78, 2162, 10.49, 144, 1.93),
    "cadmium": Element("Cd", "Cadmium", 48, 112.414, 12, 5, 321.07, 767, 8.65, 151, 1.69),
    "indium": Element("In", "Indium", 49, 114.818, 13, 5, 156.6, 2072, 7.31, 167, 1.78),
    "tin": Element("Sn", "Tin", 50, 118.71, 14, 5, 231.93, 2602, 7.31, 140, 1.96),
    "antimony": Element("Sb", "Antimony", 51, 121.76, 15, 5, 630.63, 1587, 6.697, 133, 2.05),
    "tellurium": Element("Te", "Tellurium", 52, 127.6, 16, 5, 449.51, 988, 6.24, 123, 2.1),
    "iodine": Element("I", "Iodine", 53, 126.9045, 17, 5, 113.7, 184.4, 4.93, 115, 2.5),
    "xenon": Element("Xe", "Xenon", 54, 131.293, 18, 5, -111.8, -108.1, 5.894, 108, 2.6),
    "cesium": Element("Cs", "Cesium", 55, 132.9055, 1, 6, 28.44, 671, 1.93, 265, 0.79),
    "barium": Element("Ba", "Barium", 56, 137.327, 2, 6, 727, 1897, 3.51, 222, 0.89),
    "lanthanum": Element("La", "Lanthanum", 57, 138.9055, 3, 6, 920, 3464, 6.162, 195, 1.1),
    "cerium": Element("Ce", "Cerium", 58, 140.116, 4, 6, 798, 3367, 6.77, 185, 1.12),
    "praseodymium": Element("Pr", "Praseodymium", 59, 140.9077, 4, 6, 931, 3520, 6.773, 247, 1.13),
    "neodymium": Element("Nd", "Neodymium", 60, 144.242, 4, 6, 1021, 3074, 7.01, 206, 1.14),
    "promethium": Element("Pm", "Promethium", 61, 145.0, 4, 6, 1042, 3000, 7.26, 205, 1.13),
    "samarium": Element("Sm", "Samarium", 62, 150.36, 4, 6, 1074, 1794, 7.52, 238, 1.17),
    "europium": Element("Eu", "Europium", 63, 151.964, 4, 6, 822, 1527, 5.244, 231, 1.2),
    "gadolinium": Element("Gd", "Gadolinium", 64, 157.25, 4, 6, 1313, 3273, 7.901, 233, 1.20),
    "terbium": Element("Tb", "Terbium", 65, 158.9253, 4, 6, 1356, 3230, 8.229, 225, 1.2),
    "dysprosium": Element("Dy", "Dysprosium", 66, 162.5, 4, 6, 1412, 2567, 8.55, 228, 1.22),
    "holmium": Element("Ho", "Holmium", 67, 164.9303, 4, 6, 1474, 2700, 8.795, 226, 1.23),
    "erbium": Element("Er", "Erbium", 68, 167.259, 4, 6, 1529, 2868, 9.066, 226, 1.24),
    "thulium": Element("Tm", "Thulium", 69, 168.9342, 4, 6, 1545, 1950, 9.321, 222, 1.25),
    "ytterbium": Element("Yb", "Ytterbium", 70, 173.054, 4, 6, 824, 1196, 6.965, 222, 1.1),
    "lutetium": Element("Lu", "Lutetium", 71, 174.9668, 4, 6, 1663, 3402, 9.841, 217, 1.27),
    "hafnium": Element("Hf", "Hafnium", 72, 178.49, 4, 6, 2233, 4450, 13.31, 159, 1.3),
    "tantalum": Element("Ta", "Tantalum", 73, 180.9479, 5, 6, 3017, 5458, 16.69, 143, 1.5),
    "tungsten": Element("W", "Tungsten", 74, 183.84, 6, 6, 3422, 5555, 19.25, 139, 2.36),
    "rhenium": Element("Re", "Rhenium", 75, 186.207, 7, 6, 3186, 5596, 21.02, 137, 1.9),
    "osmium": Element("Os", "Osmium", 76, 190.23, 8, 6, 3033, 5012, 22.59, 135, 2.2),
    "iridium": Element("Ir", "Iridium", 77, 192.217, 9, 6, 2446, 4428, 22.56, 136, 2.20),
    "platinum": Element("Pt", "Platinum", 78, 195.084, 10, 6, 1768.3, 3825, 21.45, 139, 2.28),
    "gold": Element("Au", "Gold", 79, 196.9665, 11, 6, 1064.18, 2856, 19.32, 144, 2.54),
    "mercury": Element("Hg", "Mercury", 80, 200.59, 12, 6, -38.83, 356.7, 13.534, 151, 2.00),
    "thallium": Element("Tl", "Thallium", 81, 204.3833, 13, 6, 303.5, 1457, 11.85, 156, 2.04),
    "lead": Element("Pb", "Lead", 82, 207.2, 14, 6, 327.46, 1749, 11.34, 154, 2.33),
    "bismuth": Element("Bi", "Bismuth", 83, 208.9804, 15, 6, 271.4, 1564, 9.78, 143, 2.02),
    "polonium": Element("Po", "Polonium", 84, 209.0, 16, 6, 254, 962, 9.196, 135, 2.0),
    "astatine": Element("At", "Astatine", 85, 210.0, 17, 6, 302, 337, 6.35, 127, 2.2),
    "radon": Element("Rn", "Radon", 86, 222.0, 18, 6, -71, -61.8, 9.73, 120, 2.2),
    "francium": Element("Fr", "Francium", 87, 223.0, 1, 7, 27, 677, 1.87, 260, 0.7),
    "radium": Element("Ra", "Radium", 88, 226.0, 2, 7, 700, 1737, 5.5, 223, 0.9),
    "actinium": Element("Ac", "Actinium", 89, 227.0, 3, 7, 1050, 3200, 10.07, 215, 1.1),
    "thorium": Element("Th", "Thorium", 90, 232.038, 4, 7, 1750, 4788, 11.72, 180, 1.3),
    "protactinium": Element("Pa", "Protactinium", 91, 231.036, 5, 7, 1572, 4000, 15.37, 180, 1.5),
    "uranium": Element("U", "Uranium", 92, 238.029, 6, 7, 1132, 3818, 19.05, 156, 1.38),
    "neptunium": Element("Np", "Neptunium", 93, 237.0, 7, 7, 644, 4000, 20.45, 155, 1.36),
    "plutonium": Element("Pu", "Plutonium", 94, 244.0, 7, 7, 639.4, 3228, 19.84, 159, 1.28),
    "americium": Element("Am", "Americium", 95, 243.0, 7, 7, 994, 2607, 13.67, 173, 1.3),
    "curium": Element("Cm", "Curium", 96, 247.0, 7, 7, 1345, 3110, 13.51, 169, 1.3),
    "berkelium": Element("Bk", "Berkelium", 97, 247.0, 7, 7, 986, 2627, 14.78, 170, 1.3),
    "californium": Element("Cf", "Californium", 98, 251.0, 7, 7, 900, 1470, 15.1, 186, 1.3),
    "einsteinium": Element("Es", "Einsteinium", 99, 252.0, 7, 7, 860, 996, 8.84, 186, 1.3),
    "fermium": Element("Fm", "Fermium", 100, 257.0, 7, 7, 1527, 827, 9.7, 200, 1.3),
    "mendelevium": Element("Md", "Mendelevium", 101, 258.0, 7, 7, 827, 0, 0, 0, 1.3),
    "nobelium": Element("No", "Nobelium", 102, 259.0, 7, 7, 827, 0, 0, 0, 1.3),
    "lawrencium": Element("Lr", "Lawrencium", 103, 262.0, 7, 7, 1627, 0, 0, 0, 1.3),
    "rutherfordium": Element("Rf", "Rutherfordium", 104, 267.0, 4, 7, 0, 0, 0, 0, 0),
    "dubnium": Element("Db", "Dubnium", 105, 270.0, 5, 7, 0, 0, 0, 0, 0),
    "seaborgium": Element("Sg", "Seaborgium", 106, 271.0, 6, 7, 0, 0, 0, 0, 0),
    "bohrium": Element("Bh", "Bohrium", 107, 270.0, 7, 7, 0, 0, 0, 0, 0),
    "hassium": Element("Hs", "Hassium", 108, 277.0, 8, 7, 0, 0, 0, 0, 0),
    "meitnerium": Element("Mt", "Meitnerium", 109, 278.0, 9, 7, 0, 0, 0, 0, 0),
    "darmstadtium": Element("Ds", "Darmstadtium", 110, 281.0, 10, 7, 0, 0, 0, 0, 0),
    "roentgenium": Element("Rg", "Roentgenium", 111, 282.0, 11, 7, 0, 0, 0, 0, 0),
    "copernicium": Element("Cn", "Copernicium", 112, 285.0, 12, 7, 0, 0, 0, 0, 0),
    "nihonium": Element("Nh", "Nihonium", 113, 286.0, 13, 7, 0, 0, 0, 0, 0),
    "flerovium": Element("Fl", "Flerovium", 114, 289.0, 14, 7, 0, 0, 0, 0, 0),
    "moscovium": Element("Mc", "Moscovium", 115, 290.0, 15, 7, 0, 0, 0, 0, 0),
    "livermorium": Element("Lv", "Livermorium", 116, 293.0, 16, 7, 0, 0, 0, 0, 0),
    "tennessine": Element("Ts", "Tennessine", 117, 294.0, 17, 7, 0, 0, 0, 0, 0),
    "oganesson": Element("Og", "Oganesson", 118, 294.0, 18, 7, 0, 0, 0, 0, 0)
}

# Allow dynamic access to elements
globals().update(periodic_table)

# Example usage
if __name__ == "__main__":
    carbon.log_usage("space simulation")
    print(f"Carbon: Weight = {carbon.atomic_weight} u, Group = {carbon.group}, Period = {carbon.period}, Boiling Point = {carbon.boiling_point} °C")
    print(f"Iron: Melting Point = {iron.melting_point} °C, Density = {iron.density} g/cm³, Atomic Radius = {iron.atomic_radius} pm")
    # Map structure using gyrogimbal
    carbon_map = carbon.map_structure(spin_rate=1.5, friction_coeff=0.15)
    print(f"Carbon structure map: {carbon_map}")
    iron_map = iron.map_structure(spin_rate=1.0, friction_coeff=0.1)
    print(f"Iron structure map: {iron_map}")
