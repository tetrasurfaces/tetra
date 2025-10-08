# haptics.py
# Copyright 2025 Beau Ayres
# Licensed under AGPL-3.0-or-later

class Haptics:
    def __init__(self):
        self.mode = "silent"
    
    def buzz(self, intensity):
        """Trigger haptic buzz for feedback (e.g., arc drift)."""
        self.mode = intensity
        print(f"Haptic buzz: {intensity}")
    
    def shake(self, intensity):
        """Trigger haptic shake for critical feedback (e.g., failure)."""
        self.mode = intensity
        print(f"Haptic shake: {intensity}")

# Module-level functions to match tetra_forge.py's expectations
def buzz(intensity):
    """Wrapper for Haptics.buzz."""
    haptics = Haptics()
    haptics.buzz(intensity)

def shake(intensity):
    """Wrapper for Haptics.shake."""
    haptics = Haptics()
    haptics.shake(intensity)

# Example usage
if __name__ == "__main__":
    buzz("low")
    shake("hard")
