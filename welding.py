# welding.py
# Copyright 2025 Beau Ayres
# Licensed under AGPL-3.0-or-later

def weave(pattern, speed, arc):
    """Execute weld weave pattern (e.g., christmas_tree)."""
    print(f"Weaving: {pattern}, speed={speed} in/min, arc={arc} mm")

def TIG(tungsten, argon, volts, hz):
    """Perform TIG welding with specified parameters."""
    print(f"TIG weld: tungsten={tungsten}, argon={argon}%, volts={volts}, freq={hz} Hz")

def acetylene(low_oxy=False, duration=None):
    """Perform acetylene cutting or marking."""
    print(f"Acetylene process: low_oxy={low_oxy}, duration={duration}s")

# Example usage
if __name__ == "__main__":
    weave("christmas_tree", 18, 1.8)
    TIG("1%lan", 98, 25, 100)
    acetylene(low_oxy=True, duration=0.8)
