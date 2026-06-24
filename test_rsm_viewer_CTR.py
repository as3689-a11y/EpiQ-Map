"""GUI-independent tests for the CTR-rod helpers.

Covers the dependency-light pieces in ``rsm_workflow`` (pair population, U
averaging, rod-config round trip) and -- when the acquisition stack is
importable -- ``autoRSM_rods`` config parsing and rod-grid construction.

Run:  conda run -n viz python -m unittest test_rsm_viewer_CTR
"""

import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np

import rsm_workflow as workflow
import Visualize_RSM_Lib as rl
from Visualize_RSM_Lib import orthonormalize, reciprocal_matrix

# The rod NeXus loaders need a recent nexusformat (with nxsetconfig, as load_rsm
# uses); skip those tests on the older acquisition-env nexusformat that lacks it.
try:
    from nexusformat.nexus import (NXdata, NXentry, NXfield, NXroot,
                                   nxsetconfig)            # noqa: F401
    HAVE_NEXUS = True
except Exception:
    HAVE_NEXUS = False

# The dialog helpers (peak_rows, compute_projection) are pure numpy but live in
# the Qt module; skip those tests where PyQt6 is not installed.
try:
    import rsm_viewer_CTR as ctr
    HAVE_CTR = True
except Exception:
    ctr = None
    HAVE_CTR = False

# The rod driver pulls in fabio/pyFAI/spec2nexus and the compiled kernel; skip
# those tests gracefully where that stack is not installed.
try:
    from HKL_Convert import autoRSM_rods            # noqa: F401
    HAVE_RODS = True
except Exception:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent / 'HKL_Convert'))
        import autoRSM_rods                          # noqa: F401
        HAVE_RODS = True
    except Exception:
        autoRSM_rods = None
        HAVE_RODS = False


def _Rz(deg):
    """Proper rotation about z by ``deg`` degrees."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _record(orientation_U, substrate='SrTiO3', lattice=(3.905,)*3 + (90.,)*3):
    Bstar = reciprocal_matrix(*lattice, with_2pi=True)
    return {
        'substrate': substrate,
        'lattice': list(map(float, lattice)),
        'orientation_U': np.asarray(orientation_U, float).tolist(),
        'Bstar': Bstar.tolist(),
        'normal': [0, 0, 1],
    }


class HKLPairsTest(unittest.TestCase):
    def test_full_product_and_order(self):
        pairs = workflow.hkl_pairs((-1, 1), (0, 1))
        self.assertEqual(
            pairs,
            [(-1, 0), (-1, 1), (0, 0), (0, 1), (1, 0), (1, 1)])

    def test_single_point(self):
        self.assertEqual(workflow.hkl_pairs((2, 2), (-1, -1)), [(2, -1)])

    def test_rejects_inverted_range(self):
        with self.assertRaises(ValueError):
            workflow.hkl_pairs((1, -1), (0, 0))


class AverageUTest(unittest.TestCase):
    def test_symmetric_average_is_proper_rotation(self):
        recs = [_record(_Rz(5)), _record(_Rz(-5))]
        meta = workflow.average_U(recs)
        U = np.asarray(meta['orientation_U'])
        # +5 and -5 about z average (after polar decomposition) back to identity.
        np.testing.assert_allclose(U, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(U)), 1.0, places=9)
        np.testing.assert_allclose(U @ U.T, np.eye(3), atol=1e-9)

    def test_matches_orthonormalized_mean_and_UB(self):
        recs = [_record(_Rz(3)), _record(_Rz(10)), _record(_Rz(-4))]
        meta = workflow.average_U(recs)
        expect = orthonormalize(np.mean([_Rz(3), _Rz(10), _Rz(-4)], axis=0))
        np.testing.assert_allclose(np.asarray(meta['orientation_U']), expect,
                                   atol=1e-9)
        Bstar = np.asarray(recs[0]['Bstar'])
        np.testing.assert_allclose(np.asarray(meta['UB']), expect @ Bstar,
                                   atol=1e-9)

    def test_rejects_mixed_substrates(self):
        recs = [_record(np.eye(3), substrate='SrTiO3'),
                _record(np.eye(3), substrate='LaAlO3')]
        with self.assertRaises(ValueError):
            workflow.average_U(recs)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            workflow.average_U([])


class WriteRodConfigTest(unittest.TestCase):
    def setUp(self):
        self.meta = workflow.average_U([_record(np.eye(3))])
        self.source = (
            'Material: STO\n'
            'Sample Name: A1\n'
            'PONI File: /cal/geo.poni\n'
            'Mask File: /cal/mask.edf\n'
            'Specfile: /spec/STO\n'
            'Temperature Directory: /raw/T300/\n'
            'Output Directory: /out/\n'
            'Scan List: [12]\n'             # stripped + replaced
            'UB: [[1,0,0],[0,1,0],[0,0,1]]\n'   # stripped + replaced
            'H Range: (-4, 4)\n')               # stripped

    def _write(self, directory):
        src = Path(directory) / 'scan12.txt'
        src.write_text(self.source)
        dest = Path(directory) / 'rods.txt'
        workflow.write_rod_config(
            str(src), str(dest), self.meta, scan_list=[12, 13],
            rod_hk_pairs=[(-1, 1), (0, 0)],
            h_window=(-0.1, 0.1), h_points=100,
            k_window=(-0.1, 0.1), k_points=100,
            l_range=(0.0, 6.0), l_points=2000,
            output_tag='rods_r01', max_intensity=1e5)
        return dest

    @staticmethod
    def _parse(text):
        cfg = {}
        for line in text.splitlines():
            if ': ' in line and not line.startswith('#'):
                key, value = line.split(': ', 1)
                cfg[key] = value
        return cfg

    def test_round_trip_keys_and_values(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._parse(self._write(d).read_text())
            # Kept beamtime keys.
            self.assertEqual(cfg['Material'], 'STO')
            self.assertEqual(cfg['Output Directory'], '/out/')
            # Replaced scan list and transfer matrix.
            self.assertEqual(ast.literal_eval(cfg['Scan List']), [12, 13])
            self.assertEqual(ast.literal_eval(cfg['Theta Scan List']), [])
            np.testing.assert_allclose(
                np.asarray(ast.literal_eval(cfg['UB'])),
                np.asarray(self.meta['UB']))
            # Rod keys.
            self.assertEqual(ast.literal_eval(cfg['Rod HK List']),
                             [(-1, 1), (0, 0)])
            self.assertEqual(ast.literal_eval(cfg['Rod H Window']), (-0.1, 0.1))
            self.assertEqual(int(cfg['Rod H Points']), 100)
            self.assertEqual(ast.literal_eval(cfg['L Range']), (0.0, 6.0))
            self.assertEqual(int(cfg['L Points']), 2000)
            self.assertEqual(cfg['Output Tag'], 'rods_r01')
            # Stripped one-off key is gone.
            self.assertNotIn('H Range', cfg)

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            dest = self._write(d)
            with self.assertRaises(FileExistsError):
                workflow.write_rod_config(
                    str(Path(d) / 'scan12.txt'), str(dest), self.meta,
                    scan_list=[12], rod_hk_pairs=[(0, 0)],
                    h_window=(-0.1, 0.1), h_points=10,
                    k_window=(-0.1, 0.1), k_points=10,
                    l_range=(0.0, 1.0), l_points=10, output_tag='x')

    def test_orientation_rotates_transfer_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 's.txt'
            src.write_text(self.source)
            workflow.write_rod_config(
                str(src), str(Path(d) / 'r.txt'), self.meta, scan_list=[12],
                rod_hk_pairs=[(0, 0)], h_window=(-0.1, 0.1), h_points=10,
                k_window=(-0.1, 0.1), k_points=10, l_range=(0.0, 1.0),
                l_points=10, output_tag='x',
                orientation=([0, 1, 0], [-1, 0, 0], [0, 0, 1]))   # 90 deg z
            cfg = self._parse((Path(d) / 'r.txt').read_text())
            expect = (np.asarray(self.meta['UB'])
                      @ workflow.orientation_matrix([0, 1, 0], [-1, 0, 0],
                                                    [0, 0, 1]))
            np.testing.assert_allclose(
                np.asarray(ast.literal_eval(cfg['UB'])), expect, atol=1e-9)

    def test_rejects_no_rods_and_bad_window(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 's.txt'
            src.write_text(self.source)
            with self.assertRaises(ValueError):
                workflow.write_rod_config(
                    str(src), str(Path(d) / 'a.txt'), self.meta,
                    scan_list=[12], rod_hk_pairs=[],
                    h_window=(-0.1, 0.1), h_points=10, k_window=(-0.1, 0.1),
                    k_points=10, l_range=(0.0, 1.0), l_points=10, output_tag='x')
            with self.assertRaises(ValueError):
                workflow.write_rod_config(
                    str(src), str(Path(d) / 'b.txt'), self.meta,
                    scan_list=[12], rod_hk_pairs=[(0, 0)],
                    h_window=(0.1, -0.1), h_points=10, k_window=(-0.1, 0.1),
                    k_points=10, l_range=(0.0, 1.0), l_points=10, output_tag='x')


@unittest.skipUnless(HAVE_RODS, 'autoRSM_rods (acquisition stack) not importable')
class RodGridTest(unittest.TestCase):
    """parse_config + build_rods contract for the rod driver."""

    def _cfg(self, directory):
        meta = workflow.average_U([_record(np.eye(3))])
        src = Path(directory) / 's.txt'
        src.write_text(
            'Material: STO\nSample Name: A1\nPONI File: /c/g.poni\n'
            'Mask File: /c/m.edf\nSpecfile: /s/STO\n'
            'Temperature Directory: /raw/\nOutput Directory: /out/\n')
        dest = Path(directory) / 'rods.txt'
        workflow.write_rod_config(
            str(src), str(dest), meta, scan_list=[12, 13],
            rod_hk_pairs=[(-1, 2), (0, 0)], h_window=(-0.1, 0.1), h_points=50,
            k_window=(-0.2, 0.2), k_points=40, l_range=(0.0, 6.0), l_points=2000,
            output_tag='rods_r01', max_intensity=1e5)
        return str(dest)

    def test_parse_and_build(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = autoRSM_rods.parse_config(self._cfg(d))
            self.assertEqual(cfg['Rod HK List'], [(-1, 2), (0, 0)])
            rods, L = autoRSM_rods.build_rods(cfg)
            self.assertEqual(len(rods), 2)
            self.assertEqual(len(L), 2000)
            np.testing.assert_allclose(L[[0, -1]], [0.0, 6.0])
            rod = rods[0]                              # (h0, k0) = (-1, 2)
            self.assertEqual((rod['h0'], rod['k0']), (-1, 2))
            self.assertEqual(len(rod['H']), 50)
            self.assertEqual(len(rod['K']), 40)
            # Window is offset from the integer center, sorted ascending.
            np.testing.assert_allclose(rod['H'][[0, -1]], [-1.1, -0.9])
            np.testing.assert_allclose(rod['K'][[0, -1]], [1.8, 2.2])
            self.assertTrue(np.all(np.diff(rod['H']) > 0))
            self.assertEqual(rod['data'].shape[0], 50 * 40 * 2000)
            self.assertTrue(np.all(rod['data'] == 0))

    def test_group_name_handles_negatives(self):
        self.assertEqual(autoRSM_rods._rod_group_name(-1, 2), 'rod_m1_2')
        self.assertEqual(autoRSM_rods._rod_group_name(0, -3), 'rod_0_m3')

    def test_kernel_clips_to_rod_and_bins(self):
        """A tight rod grid is self-clipping: HKLHIST drops pixels outside its
        bounds (other rods, out-of-L) and bins in-bounds ones correctly."""
        import hklBen
        nH = nK = 11
        nL = 21
        H = np.linspace(-0.1, 0.1, nH)
        K = np.linspace(-0.1, 0.1, nK)
        L = np.linspace(0.0, 2.0, nL)
        # M = identity, so the binned hkl equals the supplied q directly.
        pts = np.array([[0.00, 0.00, 1.0],    # inside -> binned
                        [0.05, -0.05, 0.5],   # inside -> binned
                        [0.50, 0.00, 1.0],    # H outside rod -> dropped
                        [0.00, 0.00, 5.0],    # L outside -> dropped
                        [1.00, 0.00, 1.0]])   # neighbouring rod -> dropped
        q = np.ascontiguousarray(pts.T, dtype=np.float64)
        counts = np.ones(len(pts), dtype=np.float32)
        data = np.zeros(nH * nK * nL, dtype=np.float32)
        norm = np.zeros_like(data)
        hklBen.HKLHIST(q, np.eye(3), counts, H, K, L, data, norm)
        self.assertEqual(int(norm.sum()), 2)
        filled = np.argwhere(data.reshape(nH, nK, nL) > 0).tolist()
        self.assertEqual(filled, [[5, 5, 10], [7, 2, 5]])


@unittest.skipUnless(HAVE_CTR, 'rsm_viewer_CTR (PyQt6) not importable')
class CTRHelpersTest(unittest.TestCase):
    """peak_rows and compute_projection -- the dialog's GUI-free helpers."""

    def _result(self):
        return {
            'peaks': np.array([[1.0, 0, 0], [0, 1.0, 0], [5.0, 5, 5]]),
            'hkl': np.array([[1.0, 0, 0], [0, 1.0, 0],
                             [np.nan, np.nan, np.nan]]),
            'inliers': np.array([True, True, False]),
            'U': np.eye(3),
            'Bstar': np.eye(3),
        }

    def test_peak_rows(self):
        rows = ctr.peak_rows(7, self._result())
        self.assertEqual([r['inlier'] for r in rows], [True, True, False])
        self.assertEqual(rows[0]['hkl'], (1, 0, 0))
        self.assertAlmostEqual(rows[0]['resid'], 0.0, places=6)
        self.assertAlmostEqual(rows[0]['q'], 1.0, places=6)
        self.assertIsNone(rows[2]['hkl'])
        self.assertTrue(np.isnan(rows[2]['resid']))
        self.assertEqual(rows[0]['scan'], 7)

    def test_compute_projection_shape_and_labels(self):
        axis = np.linspace(-3, 3, 31)
        data = np.zeros((31, 31, 31), np.float32)
        data[15, 15, 15] = 1.0
        result = {'peaks': np.array([[0.0, 0, 0]]),
                  'hkl': np.array([[0.0, 0, 0]]), 'inliers': np.array([True]),
                  'U': np.eye(3),
                  'Bstar': reciprocal_matrix(4, 4, 4, 90, 90, 90)}
        proj = ctr.compute_projection(data, axis, axis, axis, result,
                                      normal=[0, 0, 1], samples=(20, 20, 8))
        self.assertEqual(proj['image'].shape, (20, 20))     # (n_v, n_h)
        self.assertEqual(len(proj['h_axis']), 20)
        self.assertIn('[1 0 0]', proj['labels'][0])
        self.assertIn('[0 1 0]', proj['labels'][1])

    def test_format_direction(self):
        self.assertEqual(ctr._format_direction([1, 0, 0]), '[1 0 0]')
        self.assertEqual(ctr._format_direction([0, 0, -1]), '[0 0 -1]')


@unittest.skipUnless(HAVE_CTR, 'rsm_viewer_CTR (PyQt6) not importable')
class RodBoxTest(unittest.TestCase):
    """rod_projections / integrate_rod -- the box viewer's GUI-free helpers."""

    @staticmethod
    def _grid():
        # 4x4 in-plane cells at H = K = [0, 1, 2, 3], three L slices.
        return np.array([0., 1, 2, 3]), np.array([0., 1, 2, 3]), \
            np.array([0., 1.5, 3.0])

    def test_rod_projections_shapes_and_L_on_rows(self):
        data = np.arange(4 * 4 * 3, dtype=float).reshape(4, 4, 3)
        proj = ctr.rod_projections(data)
        self.assertEqual(proj['Z'].shape, (4, 4))      # (K, H)
        self.assertEqual(proj['HL'].shape, (3, 4))     # (L, H) -> L vertical
        self.assertEqual(proj['KL'].shape, (3, 4))     # (L, K)
        np.testing.assert_allclose(proj['Z'].T, np.nansum(data, axis=2))

    def test_integrate_full_box_matches_plain_sum(self):
        H, K, L = self._grid()
        data = np.random.default_rng(0).random((4, 4, 3))
        out_L, intensity = ctr.integrate_rod(data, H, K, L, (0, 3, 0, 3))
        np.testing.assert_allclose(intensity, np.nansum(data, axis=(0, 1)))
        np.testing.assert_allclose(out_L, L)

    def test_background_subtraction_area_normalized(self):
        H, K, L = self._grid()
        signal = np.array([10., 20., 30.])
        data = np.full((4, 4, 3), 2.0)               # flat background of 2
        data[0:2, 0:2, :] += signal                   # add signal in int box
        # int box = 2x2 cells, bkg box = 2x2 cells (same count -> ratio 1).
        _, intensity = ctr.integrate_rod(
            data, H, K, L, (0, 1, 0, 1), bkg_box=(2, 3, 2, 3))
        np.testing.assert_allclose(intensity, 4 * signal)

    def test_background_box_scaled_by_cell_ratio(self):
        H, K, L = self._grid()
        data = np.full((4, 4, 3), 5.0)               # flat -> signal is zero
        # int box 2x2 (4 cells), bkg box 1x1 (1 cell): (4/1)*5 subtracted from 4*5
        _, intensity = ctr.integrate_rod(
            data, H, K, L, (0, 1, 0, 1), bkg_box=(3, 3, 3, 3))
        np.testing.assert_allclose(intensity, np.zeros(3), atol=1e-9)

    def test_empty_box_raises(self):
        H, K, L = self._grid()
        data = np.ones((4, 4, 3))
        with self.assertRaises(ValueError):
            ctr.integrate_rod(data, H, K, L, (10, 11, 0, 1))


@unittest.skipUnless(HAVE_NEXUS, 'nexusformat not importable')
class RodLoaderTest(unittest.TestCase):
    """list_rods / load_rod / load_rsm token path for multi-rod CTR files."""

    @staticmethod
    def _axes(h0, k0):
        return (np.linspace(h0 - 0.1, h0 + 0.1, 6),
                np.linspace(k0 - 0.1, k0 + 0.1, 5),
                np.linspace(0.0, 3.0, 8))

    def _write_rods(self, path, centers):
        root = NXroot()
        root['entry'] = NXentry()
        for h0, k0 in centers:
            H, K, L = self._axes(h0, k0)
            group = NXdata(
                NXfield(np.ones((6, 5, 8), 'float32'), name='counts'),
                (NXfield(H.astype('float32'), name='H'),
                 NXfield(K.astype('float32'), name='K'),
                 NXfield(L.astype('float32'), name='L')))
            group.h0 = NXfield(np.int32(h0), name='h0')
            group.k0 = NXfield(np.int32(k0), name='k0')
            name = (f"rod_{'m' if h0 < 0 else ''}{abs(h0)}"
                    f"_{'m' if k0 < 0 else ''}{abs(k0)}")
            root['entry'][name] = group
        root.save(path)

    def test_list_and_load_token(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / 'rods.nxs')
            self._write_rods(path, [(-1, 2), (0, 0)])
            rods = rl.list_rods(path)
            self.assertEqual([(h, k) for h, k, _ in rods], [(-1, 2), (0, 0)])
            data, H, K, L = rl.load_rsm(f'{path}{rl.ROD_TOKEN}{rods[0][2]}')
            self.assertEqual(data.shape, (6, 5, 8))
            np.testing.assert_allclose([H[0], H[-1]], [-1.1, -0.9], atol=1e-4)
            np.testing.assert_allclose([L[0], L[-1]], [0.0, 3.0], atol=1e-4)
            self.assertTrue(np.allclose(np.nan_to_num(data), 1.0))

    def test_single_volume_is_not_a_rod_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / 'vol.nxs')
            H = np.linspace(-1, 1, 4).astype('float32')
            root = NXroot()
            root['entry'] = NXentry()
            root['entry']['data'] = NXdata(
                NXfield(np.ones((4, 4, 4), 'float32'), name='counts'),
                (NXfield(H, name='H'), NXfield(H, name='K'),
                 NXfield(H, name='L')))
            root.save(path)
            self.assertEqual(rl.list_rods(path), [])      # no rod_* groups
            data, _, _, _ = rl.load_rsm(path)             # plain path still works
            self.assertEqual(data.shape, (4, 4, 4))


if __name__ == '__main__':
    unittest.main()
