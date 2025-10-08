# rig.py
# Copyright 2025 Beau Ayres
# Licensed under Apache-2.0 OR AGPL-3.0-or-later

class Rig:
    def __init__(self):
        self.angle = 0
        self.torque = 0
    
    def tilt(self, direction, degrees):
        """Adjust torch or jib angle for weave or cut."""
        self.angle += degrees
        print(f"Tilted {direction} by {degrees} degrees")
    
    def stabilize(self):
        """Stabilize against crane sway or wind."""
        # Placeholder: Apply gyro corrections
        print("Stabilized rig")

# Example usage
if __name__ == "__main__":
    rig = Rig()
    rig.tilt("left", 20)
    rig.stabilize()
