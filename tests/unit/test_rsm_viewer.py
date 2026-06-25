import tempfile
import unittest
from pathlib import Path

import numpy as np

from epiq_map.rsm_viewer import (
    RegionModel, RSMViewerController, arbitrary_line_cut, axis_aligned_cut,
    intensity_view, load_region, make_region_axes, project_image, save_image,
    save_image_npz, save_region, source_tag, validate_u_matrix,
)
from epiq_map.visualize_rsm_lib import (
    axes_from_normal, reciprocal_matrix, target_orientation_matrix,
    transform_slab,
)


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


class CoAddTest(unittest.TestCase):
    """Multi-scan co-add via load_source (nanmean, identical-grid guard)."""

    def setUp(self):
        from epiq_map import rsm_viewer
        self.rv = rsm_viewer
        self.H = np.linspace(-1, 1, 4)
        a = np.full((4, 4, 4), 2.0, np.float32)
        b = np.full((4, 4, 4), 4.0, np.float32)
        b[0, 0, 0] = np.nan                       # a gap in scan b
        bad = np.ones((4, 4, 4), np.float32)
        self.store = {
            "a": (a, self.H, self.H, self.H),
            "b": (b, self.H, self.H, self.H),
            "bad": (bad, self.H, np.linspace(-2, 2, 4), self.H),  # different K
        }
        self._orig = rsm_viewer.load_rsm
        rsm_viewer.load_rsm = lambda p: self.store[str(p)]
        self.ctrl = RSMViewerController(object())

    def tearDown(self):
        self.rv.load_rsm = self._orig

    def test_nanmean_and_nan_gap(self):
        self.ctrl.load_source(["a", "b"])
        self.assertAlmostEqual(self.ctrl.source_data[1, 1, 1], 3.0)   # (2+4)/2
        self.assertAlmostEqual(self.ctrl.source_data[0, 0, 0], 2.0)   # b is NaN -> just a
        self.assertEqual(len(self.ctrl.source_paths), 2)

    def test_needs_reload(self):
        self.ctrl.load_source(["a", "b"])
        self.assertFalse(self.ctrl.needs_reload(["a", "b"]))
        self.assertTrue(self.ctrl.needs_reload(["a"]))

    def test_mismatched_grid_rejected(self):
        with self.assertRaises(ValueError):
            self.ctrl.load_source(["a", "bad"])


class ProjectImageTest(unittest.TestCase):
    """2D slab projection used by the RSM image dock."""

    def setUp(self):
        ax = np.linspace(-1, 1, 9)
        H, K, L = np.meshgrid(ax, ax, ax, indexing="ij")
        self.data = (H + 2 * K + 3 * L).astype("float32")   # linear field
        self.axes = (ax, ax, ax)

    def test_slab_mean_projects_inplane_field(self):
        # In-plane H (horiz), K (vert); average over a thin L slab -> H + 2K.
        img, h, v = project_image(self.data, self.axes, np.eye(3),
                                  ((-0.5, 0.5), (-0.5, 0.5)), (5, 4),
                                  0.0, 0.5, 5, "mean")
        self.assertEqual(img.shape, (4, 5))                  # (n_v, n_h)
        expect = h[None, :] + 2 * v[:, None]
        np.testing.assert_allclose(img, expect, atol=1e-6)

    def test_single_slice(self):
        img, h, v = project_image(self.data, self.axes, np.eye(3),
                                  ((-0.5, 0.5), (-0.5, 0.5)), (5, 4),
                                  0.0, 0.0, 1, "mean")
        np.testing.assert_allclose(img, h[None, :] + 2 * v[:, None], atol=1e-6)

    def test_sum_is_n_times_mean(self):
        mean, _, _ = project_image(self.data, self.axes, np.eye(3),
                                   ((-.5, .5), (-.5, .5)), (5, 4), 0.0, 0.5, 5, "mean")
        ssum, _, _ = project_image(self.data, self.axes, np.eye(3),
                                   ((-.5, .5), (-.5, .5)), (5, 4), 0.0, 0.5, 5, "sum")
        np.testing.assert_allclose(ssum, 5 * mean, atol=1e-4)

    def test_export_png_and_npz(self):
        img, h, v = project_image(self.data, self.axes, np.eye(3),
                                  ((-.5, .5), (-.5, .5)), (5, 4), 0.0, 0.5, 5, "mean")
        with tempfile.TemporaryDirectory() as d:
            png = Path(d) / "img.png"
            save_image(str(png), img, h, v, cmap="inferno", labels=("H", "K"))
            self.assertTrue(png.exists() and png.stat().st_size > 0)
            npz = Path(d) / "img.npz"
            save_image_npz(str(npz), img, h, v, {"unit": "A^-1"})
            with np.load(str(npz), allow_pickle=True) as z:
                np.testing.assert_array_equal(z["image"], img)
                np.testing.assert_allclose(z["h_axis"], h)

    def test_bad_reduction_rejected(self):
        with self.assertRaises(ValueError):
            project_image(self.data, self.axes, np.eye(3),
                          ((-.5, .5), (-.5, .5)), (5, 4), 0.0, 0.5, 5, "median")


class SourceTagTest(unittest.TestCase):
    def test_prefers_trailing_number(self):
        self.assertEqual(source_tag("/data/scan_0042.nxs"), "0042")
        self.assertEqual(source_tag("foo_007.nxs"), "007")

    def test_falls_back_to_stem(self):
        self.assertEqual(source_tag("rsm.nxs"), "rsm")
        self.assertEqual(source_tag(""), "?")


if __name__ == "__main__":
    unittest.main()
