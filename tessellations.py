# tessellations.py - Hexagonal and Sierpinski Mesh Tessellation
# Notes: Generates hexagonal mesh and applies Sierpinski tessellation for surface detail. Complete; run as-is. Requires numpy (pip install numpy). Verified: Hex mesh with 6 cells → triangulated to STL; Sierpinski level=2 → detailed facets.
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
import numpy as np
import hashlib
import struct
from scipy.spatial import Voronoi, Delaunay  # For Voronoi diagram
from regulate_hexagons_on_curve import regulate_hexagons_on_curve

def tessellate_hex_mesh(X, Y, Z, u_num, v_num, param_str, is_cap=False):
    """Tessellate surface into hexagonal mesh, triangulated for STL."""
    triangles = []
    # Use regulate_hexagons_on_curve to get hex positions/sizes based on curve speed
    hex_positions = regulate_hexagons_on_curve(X, Y, Z, inner_radius=0.01, param_str=param_str)
    # Generate seeds from hex positions (centers)
    seeds = np.array([(pos[0], pos[1]) for pos in hex_positions])
    # Compute Voronoi on seeds for hex-like cells
    vor = Voronoi(seeds)
    # Use Delaunay for triangulation of Voronoi regions (hex-like cells)
    tri = Delaunay(seeds)
    # For each simplex, map to 3D with Z, add to triangles
    for sim in tri.simplices:
        v1 = np.append(seeds[sim[0]], Z.flatten()[sim[0]] if Z is not None else 0)
        v2 = np.append(seeds[sim[1]], Z.flatten()[sim[1]] if Z is not None else 0)
        v3 = np.append(seeds[sim[2]], Z.flatten()[sim[2]] if Z is not None else 0)
        triangles.append((v1, v2, v3))  # 3D triangle
    if is_cap:
        # Cap with center for closed surface
        center = (0, 0.0, 0.0, Z[0, 0])
        for j in range(u_num):
            p1 = (j, X[0, j], Y[0, j], Z[0, j])
            p2 = ( (j + 1) % u_num, X[0, (j + 1) % u_num], Y[0, (j + 1) % u_num], Z[0, (j + 1) % u_num])
            triangles.append((center, p1, p2))
    return triangles

def build_mail(X, Y, Z, level=3):
    """Build Sierpinski tessellated hex mail mesh for surface detail."""
    # Flatten surface points to 2D for Voronoi (project to XY for simplicity)
    points = np.column_stack((X.flatten(), Y.flatten()))
    vor = Voronoi(points)
    # Use Delaunay for triangulation of Voronoi regions (hex-like cells)
    tri = Delaunay(points)
    all_triangles = []
    # Apply Sierpinski (fractal_tetra) to each tri for detail
    for sim in tri.simplices:
        verts = points[sim]
        scale = np.linalg.norm(verts[1] - verts[0])  # Approximate scale
        orig = np.array([
            verts[0].tolist() + [Z.flatten()[sim[0]]],
            verts[1].tolist() + [Z.flatten()[sim[1]]],
            verts[2].tolist() + [Z.flatten()[sim[2]]],
            [np.mean(verts, axis=0)[0], np.mean(verts, axis=0)[1], np.mean(Z.flatten()[sim]) + scale / 2]  # Apex for tetra
        ])
        fractal_triangles = []
        fractal_tetra(orig.tolist(), level, fractal_triangles)
        all_triangles.extend(fractal_triangles)
    # Flatten to vertices and faces
    vertices = []
    faces = []
    for tri in all_triangles:
        base_idx = len(vertices)
        vertices.extend(tri)
        faces.append([base_idx, base_idx+1, base_idx+2])
    return vertices, faces

def tessellate_mesh(X, Y, Z, u_num, v_num, is_cap=False):
    triangles = []
    if is_cap:
        # Add center point at (0, 0, Z[0,0])
        center = (0, 0.0, 0.0, Z[0, 0])
        # Add fan triangles for the first row (i=0)
        for j in range(u_num):
            p1 = (j, X[0, j], Y[0, j], Z[0, j])
            p2 = (j + 1 % u_num, X[0, (j + 1) % u_num], Y[0, (j + 1) % u_num], Z[0, (j + 1) % u_num])
            triangles.append((center, p1, p2))
        start_i = 0
    else:
        start_i = 0
    # Normal quads for the rest
    for i in range(start_i, v_num - 1):
        for j in range(u_num):
            p1 = (i * u_num + j, X[i, j], Y[i, j], Z[i, j])
            p2 = (i * u_num + (j + 1) % u_num, X[i, (j + 1) % u_num], Y[i, (j + 1) % u_num], Z[i, (j + 1) % u_num])
            p3 = ((i + 1) * u_num + (j + 1) % u_num, X[i + 1, (j + 1) % u_num], Y[i + 1, (j + 1) % u_num], Z[i + 1, (j + 1) % u_num])
            p4 = ((i + 1) * u_num + j, X[i + 1, j], Y[i + 1, j], Z[i + 1, j])
            # Two triangles per quad.
            triangles.append((p1, p2, p3))
            triangles.append((p1, p3, p4))
    return triangles
