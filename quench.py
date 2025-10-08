# quench.py
# Copyright 2025 Beau Ayres
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

# Adaptation: Add oil quench logging
def log_quench(temp_profile, porosity_threshold=0.2):
    """
    Logs temperature-drop profile during oil quenching and estimates porosity.
    - temp_profile: List of temperatures over time.
    - porosity_threshold: Threshold for void bloating (default 0.2 for 20% voids).
    """
    for t, temp in enumerate(temp_profile):
        log(f"Time {t}: Temp {temp}C")
        if temp < 200:
            voids = (200 - temp) / 200 * porosity_threshold  # Simplified void estimation
            log(f"Void growth: {voids * 100:.1f}%")
    print("Quench log complete")

# Example usage
if __name__ == "__main__":
    mock_temp = [900, 700, 500, 300, 100, 20]  # Simulated temp drop
    log_quench(mock_temp)
