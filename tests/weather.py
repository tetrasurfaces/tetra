# weather.py
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
import time

class Wind:
    """Simulate dynamic wind conditions for meteorological integration."""
    def __init__(self, base_speed=5.0, base_direction=0.0, variation_rate=0.1):
        self.base_speed = base_speed  # Base wind speed in m/s
        self.base_direction = base_direction  # Base wind direction in degrees
        self.variation_rate = variation_rate  # Rate of change per second
        self.last_update = time.time()
        self.current_speed = base_speed
        self.current_direction = base_direction

    def update(self):
        """Update wind conditions based on time elapsed."""
        current_time = time.time()
        elapsed = current_time - self.last_update
        self.last_update = current_time
        # Random variation in speed and direction
        speed_variation = np.random.normal(0, self.variation_rate * elapsed)
        direction_variation = np.random.normal(0, 5.0 * self.variation_rate * elapsed)
        self.current_speed = np.clip(self.base_speed + speed_variation, 0.0, 20.0)
        self.current_direction = np.mod(self.base_direction + direction_variation, 360.0)
        # Simulate gust with 10% chance
        if np.random.random() < 0.1:
            gust = np.random.uniform(0.0, 5.0)
            self.current_speed += gust

    def get_wind(self):
        """Get current wind speed and direction."""
        self.update()
        return self.current_speed, self.current_direction

    def simulate_gust(self, intensity=5.0):
        """Simulate a gust event."""
        self.current_speed += intensity
        self.current_speed = np.clip(self.current_speed, 0.0, 25.0)  # Cap at 25 m/s for extreme gusts
        print(f"Gust detected: Speed increased to {self.current_speed} m/s")

if __name__ == "__main__":
    # Example usage
    wind = Wind(base_speed=5.0, base_direction=45.0)
    speed, direction = wind.get_wind()
    print(f"Initial wind: Speed = {speed} m/s, Direction = {direction} degrees")
    wind.simulate_gust(intensity=3.0)
    speed, direction = wind.get_wind()
    print(f"After gust: Speed = {speed} m/s, Direction = {direction} degrees")
