# haptics.py
# Copyright 2025 Beau Ayres
# Licensed under Apache-2.0 OR AGPL-3.0-or-later

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

# Example usage
if __name__ == "__main__":
    haptic = Haptics()
    haptic.buzz("low")
    haptic.shake("hard")
