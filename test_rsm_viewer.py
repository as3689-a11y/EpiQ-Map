import tempfile
import unittest
from pathlib import Path

import numpy as np

from rsm_viewer import (RegionModel, RSMViewerController, arbitrary_line_cut,
                        axis_aligned_cut, intensity_view, load_region,
                        make_region_axes, save_region, source_tag,
                        validate_u_matrix)
from Visualize_RSM_Lib import (axes_from_normal, reciprocal_matrix,
                               target_orientation_matrix, transform_slab)


class RSMViewerHelpersTest(unittest.TestCase):
    def setUp(self):
        self.axes = tuple(np.linspace(-1, 1, 9) for _ in range(3))
        H, K, L = np.meshgrid(*self.axes, indexing="ij")
        self.model = RegionModel((H + 2*K + 3*L).astype("float32"), self.axes)

    def test_axis_cut_known_linear_volume(self):
        x, y = axis_aligned_cut(self.model, 2, (0, 0), (0, 0))
        np.testing.assert_allclose(y, 3*x, atol=1e-6)

    def test_arbitrary_line_known_linear_volume(self):
        x, y = arbitrary_line_cut(self.model, [-1, 0, 0], [1, 0, 0], 9)
        np.testing.assert_allclose(x, np.linspace(0, 2, 9))
        np.testing.assert_allclose(y, np.linspace(-1, 1, 9), atol=1e-6)

    def test_small_identity_interpolation(self):
        target = tuple(np.linspace(-.5, .5, 5) for _ in range(3))
        out = transform_slab(self.model.volume, *self.axes, np.eye(3), *target)
        H, K, L = np.meshgrid(*target, indexing="ij")
        np.testing.assert_allclose(out, H + 2*K + 3*L, atol=1e-6)

    def test_npz_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "region.npz"
            save_region(str(path), self.model)
            loaded = load_region(str(path))
            np.testing.assert_array_equal(loaded.volume, self.model.volume)
            np.testing.assert_array_equal(loaded.U, np.eye(3))

    def test_validation_and_log(self):
        with self.assertRaises(ValueError): validate_u_matrix(np.eye(2))
        with self.assertRaises(ValueError): make_region_axes([(1, 0)]*3, (2,2,2))
        out = intensity_view(np.array([-2, 0, 3, np.nan], np.float32), "log1p")
        np.testing.assert_allclose(out, [0, 0, np.log(4), 0])


class NormalToAxesTest(unittest.TestCase):
    def setUp(self):
        self.Bs = reciprocal_matrix(4, 4, 4, 90, 90, 90)        # cubic

    def _cartesian(self, hkl):
        v = self.Bs @ np.asarray(hkl, float)
        return v / np.linalg.norm(v)

    def test_001_gives_canonical_frame(self):
        ox, oy, oz = axes_from_normal(self.Bs, [0, 0, 1])
        np.testing.assert_array_equal(ox, [1, 0, 0])
        np.testing.assert_array_equal(oy, [0, 1, 0])
        np.testing.assert_array_equal(oz, [0, 0, 1])

    def test_orthonormal_right_handed_and_z_is_normal(self):
        for normal in ([0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 1]):
            ox, oy, oz = axes_from_normal(self.Bs, normal)
            cx, cy, cz = (self._cartesian(d) for d in (ox, oy, oz))
            for a, b in ((cx, cy), (cx, cz), (cy, cz)):
                self.assertAlmostEqual(float(np.dot(a, b)), 0.0, places=6)
            self.assertAlmostEqual(np.linalg.det([cx, cy, cz]), 1.0, places=6)
            self.assertAlmostEqual(abs(np.dot(cz, self._cartesian(normal))), 1.0, places=6)
            # the triple must satisfy target_orientation_matrix's 90-deg gate
            target_orientation_matrix(self.Bs, ox, oy, oz)

    def test_deterministic(self):
        a = axes_from_normal(self.Bs, [1, 1, 0])
        b = axes_from_normal(self.Bs, [1, 1, 0])
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_tetragonal_normal_out_of_plane(self):
        Bt = reciprocal_matrix(4, 4, 6, 90, 90, 90)
        _, _, oz = axes_from_normal(Bt, [0, 0, 1])
        cz = Bt @ oz; cz /= np.linalg.norm(cz)
        zn = Bt @ np.array([0, 0, 1.0]); zn /= np.linalg.norm(zn)
        self.assertAlmostEqual(abs(np.dot(cz, zn)), 1.0, places=6)

    def test_rejects_bad_normal(self):
        with self.assertRaises(ValueError):
            axes_from_normal(self.Bs, [0, 0, 0])


class DisplayScaleTest(unittest.TestCase):
    """Equal-axes (cube) rendering scale -- pure numeric, no Qt needed."""

    def setUp(self):
        axes = (np.linspace(0, 6, 16),       # span 6
                np.linspace(-0.3, 0.3, 32),  # span 0.6
                np.linspace(-4, 4, 64))      # span 8
        self.model = RegionModel(np.zeros((16, 32, 64), "float32"), axes)
        self.ctrl = RSMViewerController(object())

    def _extents(self, scale):
        return [s * (n - 1) for s, n in zip(scale, self.model.volume.shape)]

    def test_natural_keeps_true_proportions(self):
        self.ctrl.equal_axes = False
        np.testing.assert_allclose(self.ctrl.display_scale(self.model), self.model.scale)
        np.testing.assert_allclose(self._extents(self.ctrl.display_scale(self.model)),
                                   [6.0, 0.6, 8.0])

    def test_equal_axes_makes_a_cube(self):
        self.ctrl.equal_axes = True
        ext = self._extents(self.ctrl.display_scale(self.model))
        np.testing.assert_allclose(ext, [8.0, 8.0, 8.0])   # target = max span


class SourceTagTest(unittest.TestCase):
    def test_prefers_trailing_number(self):
        self.assertEqual(source_tag("/data/scan_0042.nxs"), "0042")
        self.assertEqual(source_tag("foo_007.nxs"), "007")

    def test_falls_back_to_stem(self):
        self.assertEqual(source_tag("rsm.nxs"), "rsm")
        self.assertEqual(source_tag(""), "?")


if __name__ == "__main__":
    unittest.main()
