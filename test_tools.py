# test_tools.py
# Copyright 2025 Beau Ayres
# Licensed under Apache-2.0 OR AGPL-3.0-or-later

def flex_until_break(load, ram):
    """Simulate flex test until failure."""
    print(f"Flex test: load={load}, ram={ram}")
    # Placeholder: Return False for no failure (to resolve failure variable)
    return False

def ink_test(dye, uv):
    """Perform dye penetration test for defects."""
    print(f"Ink test: dye={dye}, uv={uv}")

# Example usage
if __name__ == "__main__":
    flex_until_break("5mm/min", "hydraulic")
    ink_test("red_dye", True)
