# telemetry_nav.py
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

import hashlib

def log_ipfs_navigation(vector_data, cache_moves=5):
    """
    Logs supply chain vector navigation using IPFS-style hashing.
    - vector_data: List of (position, speed, load) tuples for fleet vectors.
    - cache_moves: Number of moves to cache locally (default 5).
    Returns: List of IPFS-style hashes for navigation steps.
    """
    ipfs_hashes = []
    for i, (pos, speed, load) in enumerate(vector_data):
        # Create a content-addressed hash for each move
        move_str = f"{pos}_{speed}_{load}"
        move_hash = hashlib.sha256(move_str.encode()).hexdigest()
        ipfs_hashes.append(move_hash)
        log(f"Move {i}: Pos {pos}, Speed {speed}, Load {load}, Hash {move_hash[:8]}")
        
        # Cache up to cache_moves
        if i >= cache_moves:
            log(f"Evicting oldest move: {ipfs_hashes[i - cache_moves][:8]}")
            ipfs_hashes.pop(0)
    
    return ipfs_hashes

# Example usage
if __name__ == "__main__":
    mock_vectors = [(0, 10, 100), (1, 12, 95), (2, 11, 98), (3, 9, 102), (4, 10, 100)]
    hashes = log_ipfs_navigation(mock_vectors)
    print(f"IPFS navigation hashes: {hashes}")
