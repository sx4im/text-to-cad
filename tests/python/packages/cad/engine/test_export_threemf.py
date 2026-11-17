"""A 3MF export's vertices (`export/threemf.py`, `export/mesh.py`): one per position (#843).

A part is meshed face by face, so a corner shared by three faces arrives as three
positions; the writer used to emit them verbatim, and a slicer counts an edge by index,
so every seam edge read as an open one (a 20 mm cube exported 24 vertices, 24 edges used
once). The writer merges positions within 1e-5 mm per object and remaps the triangles,
so the cube exports its 8 corners and every edge is used by exactly two triangles.
"""

from __future__ import annotations

import re
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from cad.engine.export import Placed, Source, View
from cad.engine.export import threemf
from cad.engine.export.mesh import merged
from cad.engine.mesh_format import Mesh


def _cube_mesh() -> Mesh:
    """A 20 mm cube meshed the way the tessellator does: each face's own copies of its corners."""
    corners = np.array(
        [[0, 0, 0], [20, 0, 0], [20, 20, 0], [0, 20, 0],
         [0, 0, 20], [20, 0, 20], [20, 20, 20], [0, 20, 20]], dtype=np.float64)
    quads = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
             (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)]
    positions, indices, faces = [], [], []
    for ordinal, (a, b, c, d) in enumerate(quads):
        first = len(indices)
        base = len(positions)
        positions.extend(corners[[a, b, c, d]])
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
        faces.append((ordinal, first, 6))
    return Mesh(
        np.array(positions, dtype=np.float32),
        np.zeros((len(positions), 3), dtype=np.float32),
        np.array(indices, dtype=np.uint32),
        np.array(faces, dtype=np.uint32),
        np.zeros((0, 4), dtype=np.uint32),
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 4), dtype=np.float32),
    )


def _write(mesh: Mesh, folder: Path, *, face_colors: dict | None = None) -> Path:
    prototype: dict = {"key": "cube", "name": "cube"}
    if face_colors is not None:
        prototype["faceColors"] = face_colors
    structure = {
        "name": "cube",
        "prototypes": [prototype],
        "instances": [{"prototype": 0, "matrix": list(np.eye(4).T.reshape(-1)), "path": "cube"}],
        "tree": {"instance": 0, "name": "cube"},
    }
    source = Source(structure, {}, live=False, meshes={"cube": mesh})
    view = View(source, "cube", [Placed(0, np.eye(4), ["cube"], None, 0)], {"instance": 0}, [np.eye(4)])
    out = folder / "cube.3mf"
    threemf.write(view, out)
    return out


def _edge_use(path: Path) -> tuple[int, Counter]:
    with zipfile.ZipFile(path) as archive:
        model = archive.read("3D/3dmodel.model").decode()
    vertices = len(re.findall(r"<vertex ", model))
    triangles = re.findall(r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"', model)
    use: Counter = Counter()
    for a, b, c in triangles:
        for edge in (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))):
            use[edge] += 1
    return vertices, use


class MergedTest(unittest.TestCase):
    def test_an_already_unique_mesh_comes_back_as_itself(self) -> None:
        mesh = Mesh(
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
            np.array([0, 1, 2], dtype=np.uint32),
            np.array([(0, 0, 3)], dtype=np.uint32),
            np.zeros((0, 4), dtype=np.uint32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )
        self.assertIs(merged(mesh), mesh)

    def test_nearby_positions_merge_and_distant_ones_do_not(self) -> None:
        mesh = Mesh(
            np.array([[0, 0, 0], [0.000003, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            np.zeros((4, 3), dtype=np.float32),
            np.array([0, 2, 3, 1, 2, 3], dtype=np.uint32),
            np.array([(0, 0, 3), (1, 3, 3)], dtype=np.uint32),
            np.zeros((0, 4), dtype=np.uint32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )
        found = merged(mesh)
        self.assertEqual(len(found.positions), 3, "the 3e-6 mm pair is one vertex")
        self.assertEqual(found.indices.tolist(), [0, 1, 2, 0, 1, 2])
        self.assertEqual(found.faces.tolist(), mesh.faces.tolist(), "firsts and counts are unchanged")

    def test_an_empty_mesh_comes_back_as_itself(self) -> None:
        mesh = Mesh(
            np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.uint32), np.zeros((0, 3), dtype=np.uint32),
            np.zeros((0, 4), dtype=np.uint32), np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )
        self.assertIs(merged(mesh), mesh)


class ThreemfTest(unittest.TestCase):
    def test_a_cube_exports_eight_vertices_with_every_edge_used_twice(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            vertices, use = _edge_use(_write(_cube_mesh(), Path(folder)))
        self.assertEqual(vertices, 8, "one vertex per corner, not three")
        self.assertEqual(len(use), 18, "12 cube edges plus 6 face diagonals")
        self.assertTrue(all(count == 2 for count in use.values()),
                        "every edge is shared by exactly two triangles")

    def test_a_painted_face_still_exports_merged(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            vertices, use = _edge_use(_write(_cube_mesh(), Path(folder),
                                             face_colors={"0": (1.0, 0.0, 0.0, 1.0)}))
        self.assertEqual(vertices, 8)
        self.assertTrue(all(count == 2 for count in use.values()))


if __name__ == "__main__":
    unittest.main()
