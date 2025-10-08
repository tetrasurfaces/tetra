# maptics.py
# Copyright 2025 Beau Ayres
# Licensed under Apache-2.0 OR AGPL-3.0-or-later

class Maptics:
    def __init__(self):
        self.path = []
    
    def record_path(self, x, y, z, angle):
        """Record welder's hand or torch path."""
        self.path.append((x, y, z, angle))
        print(f"Path recorded: {(x, y, z, angle)}")
    
    def replay_path(self):
        """Replay ghosted hand path for training or automation."""
        print("Replaying path:", self.path)

# Example usage
if __name__ == "__main__":
    maptics = Maptics()
    maptics.record_path(0, 0, 0, 15)
    maptics.replay_path()
