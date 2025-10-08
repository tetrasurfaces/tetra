# prep_tools.py
# Copyright 2025 Beau Ayres
# Licensed under Apache-2.0 OR AGPL-3.0-or-later

def angle_grinder(angle, rpm, coolant):
    """Simulate angle grinding for surface prep."""
    print(f"Grinding: angle={angle} deg, rpm={rpm}, coolant={coolant}")

def swarf_vacuum():
    """Remove swarf and sparks from milling/grinding."""
    print("Swarf vacuumed")

def acetylene_mark(low_oxy, duration):
    """Apply carbon marking with acetylene torch."""
    print(f"Marking: low_oxy={low_oxy}, duration={duration}s")

def auto_markup(angle1, angle2, root_gap):
    """Automatic markup for V bevel."""
    print(f"Markup: angles=({angle1}, {angle2}) deg, root_gap={root_gap} mm")

# Example usage
if __name__ == "__main__":
    angle_grinder(30, 20000, "water")
    swarf_vacuum()
    acetylene_mark(True, 0.8)
    auto_markup(30, 45, 2.0)
