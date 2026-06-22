import json
import os
import sys
import tempfile
import unittest

import numpy as np

import rsm_monitor
import rsm_workflow as workflow

HERE = os.path.dirname(os.path.abspath(__file__))
LATTICES = os.path.join(HERE, 'substrate_lattice_constants.txt')

# The acquisition stack (autoRSM + hklBen + libhklBen.so) lives in HKL_Convert,
# populated separately. Tests that need it are skipped until it is in place.
sys.path.insert(0, os.path.join(HERE, 'HKL_Convert'))
try:
    import autoRSM
except ImportError:
    autoRSM = None

requires_autorsm = unittest.skipIf(
    autoRSM is None, 'autoRSM acquisition stack not present in HKL_Convert/')


class Wrapper3Tests(unittest.TestCase):
    def test_lattice_entries_and_direction_parser(self):
        entries = workflow.load_lattice_entries(LATTICES)
        self.assertIn('LaAlO3', entries)
        self.assertIn('TiO2', entries)
        self.assertEqual(workflow.parse_direction('[1, -1, 0]'),
                         [1.0, -1.0, 0.0])

    def test_index_metadata_ub_maps_hkl_to_measured_q(self):
        lattice = workflow.rl.load_lattice(LATTICES, 'LaAlO3')
        Bstar = workflow.rl.reciprocal_matrix(*lattice)
        rotation = workflow.rl.Rotation.from_euler(
            'xyz', [0.04, -0.03, 0.08]).as_matrix()
        hkl = np.array([
            [-2, -1, 1], [-2, 0, 1], [-1, -2, 2], [-1, 0, 1],
            [0, -2, 1], [0, -1, 2], [1, 0, 1], [1, 1, 2],
            [2, -1, 1], [2, 0, 2], [1, -2, 1], [-2, 1, 2],
        ], dtype=float)
        peaks = (rotation @ Bstar @ hkl.T).T
        result = workflow.index_with_substrate(
            None, None, None, None, LATTICES, 'LaAlO3', peaks=peaks)
        self.assertIsNotNone(result)
        metadata = workflow.build_index_metadata(
            result, 'LaAlO3', lattice, '/tmp/source.nxs')
        UB = np.asarray(metadata['UB'])
        assigned = result['hkl'][result['inliers']]
        predicted = (UB @ assigned.T).T
        np.testing.assert_allclose(
            predicted, peaks[result['inliers']], atol=0.05)

    @requires_autorsm
    def test_reconstruction_config_and_custom_grid(self):
        metadata = {
            'UB': np.eye(3).tolist(),
            'lattice': [3.79, 3.79, 3.79, 90, 90, 90],
        }
        base = """PONI File: geometry.poni
Material: Test
Sample Name: sample
Scan Number: 12
Scan List: [12]
Theta Scan List: []
Temperature: 100
Mask File: mask.edf
Specfile: spec.dat
Temperature Directory: /tmp/data/
Image Directory: /tmp/data/Test_012/
Output Directory: /tmp/output/
"""
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, 'source.txt')
            derived = os.path.join(tmp, 'derived.txt')
            with open(source, 'w') as fh:
                fh.write(base)
            workflow.write_reconstruction_config(
                source, derived, metadata,
                {'H': (-2, 2), 'K': (-0.3, 0.3), 'L': (0, 5)},
                (101, 31, 151), 'indexed_LAO_r01')
            cfg = autoRSM.parse_config(derived)
            H, K, L, UB, out = autoRSM.build_grids(cfg, None)
            self.assertEqual((len(H), len(K), len(L)), (101, 31, 151))
            self.assertEqual((H[0], H[-1]), (-2, 2))
            np.testing.assert_allclose(UB, np.eye(3))
            self.assertTrue(out.endswith('indexed_objects'))
            # The Output Tag carries the audit tag + H/K/L range so each
            # indexed reconstruction of a scan lands in its own named .nxs.
            self.assertEqual(cfg['Output Tag'],
                             'indexed_LAO_r01_H-2to2_K-0.3to0.3_L0to5')
            self.assertEqual(
                autoRSM.output_filename(cfg),
                'Test_sample_scans_12_indexed_LAO_r01_H-2to2_K-0.3to0.3_L0to5.nxs')

    def test_metadata_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'scan_U.txt')
            metadata = {'U': np.eye(3).tolist(), 'substrate': 'test'}
            workflow.save_index_metadata(path, metadata)
            np.testing.assert_allclose(np.loadtxt(path), np.eye(3))
            self.assertEqual(workflow.load_index_metadata(path), metadata)
            with self.assertRaises(FileExistsError):
                workflow.save_index_metadata(path, metadata)

    def test_optional_scaled_ub_s_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'scan_U_S.txt')
            metadata = {
                'U': np.eye(3).tolist(),
                'UB': (2 * np.eye(3)).tolist(),
                'substrate': 'test',
                'save_scaled_ub': True,
            }
            workflow.save_index_metadata(path, metadata)
            ub_path = os.path.join(tmp, 'scan_UB_S.txt')
            np.testing.assert_allclose(np.loadtxt(ub_path), 2 * np.eye(3))

    def test_dataset_uses_versioned_u_s_without_touching_legacy_u(self):
        with tempfile.TemporaryDirectory() as tmp:
            transformed = os.path.join(tmp, 'transformed_objects')
            os.makedirs(transformed)
            output = os.path.join(transformed, 'Mat_sample_scans_7_out.nxs')
            open(output, 'w').close()
            legacy = output[:-4] + '_U.txt'
            np.savetxt(legacy, np.eye(3))
            config = os.path.join(tmp, 'scan.txt')
            with open(config, 'w') as fh:
                fh.write('Material: Mat\nSample Name: sample\nScan List: [7]\n'
                         'Theta Scan List: []\nOutput Directory: ' + tmp + '\n')
            dataset = rsm_monitor.Dataset(config)
            self.assertTrue(dataset.next_u_path().endswith('_U_S.txt'))
            metadata = {'U': np.eye(3).tolist(), 'substrate': 'test'}
            workflow.save_index_metadata(dataset.next_u_path(), metadata)
            self.assertTrue(os.path.exists(legacy))
            self.assertTrue(dataset.u_path().endswith('_U_S.txt'))
            workflow.save_index_metadata(dataset.next_u_path(), metadata)
            self.assertTrue(dataset.u_path().endswith('_U_S_02.txt'))

    @requires_autorsm
    def test_spec_reader_matches_autorsm_usage(self):
        # autoRSM reads each scan through spec2nexus' SpecDataFile, then pulls
        # per-frame columns from scan.data and fixed angles from
        # scan.positioner (see transform_scan / theta_scan). Guard exactly
        # that interface against a synthetic spec file.
        content = """#F synthetic.dat
#E 1
#D Wed Mar 25 12:00:00 2026
#O0 th  chi  phi

#S 12 ascan phi 0 1 2 1
#D Wed Mar 25 12:00:00 2026
#P0 1.5 2.5 3.5
#N 3
#L phi  ic2  detector
0.0 10 1
0.5 11 2
1.0 12 3
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'spec.dat')
            with open(path, 'w') as fh:
                fh.write(content)
            scan = autoRSM.SpecDataFile(path).getScan(12)
            np.testing.assert_allclose(np.asarray(scan.data['phi']), [0, 0.5, 1])
            np.testing.assert_allclose(np.asarray(scan.data['ic2']), [10, 11, 12])
            self.assertEqual(float(scan.positioner['chi']), 2.5)
            self.assertEqual(float(scan.positioner['th']), 1.5)

    def test_original_server_config_has_only_supported_keys(self):
        metadata = {
            'UB': (2 * np.eye(3)).tolist(),
            'lattice': [3.79, 3.79, 3.79, 90, 90, 90],
            'u_s_record': '/tmp/scan_U_S.json',
        }
        base = """PONI File: geometry.poni
Material: Test
Sample Name: sample
Scan List: [12]
Theta Scan List: []
Mask File: mask.edf
Specfile: spec.dat
Temperature Directory: /tmp/data/
Output Directory: /tmp/output/
"""
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, 'source.txt')
            derived = os.path.join(tmp, 'derived.txt')
            with open(source, 'w') as fh:
                fh.write(base)
            workflow.write_reconstruction_config(
                source, derived, metadata,
                {'H': (-2, 2), 'K': (-2, 2), 'L': (0, 5)},
                (20, 20, 20), 'ignored', custom_grid=False)
            with open(derived) as fh:
                text = fh.read()
            self.assertIn('UB: ', text)
            self.assertIn('Substrate Lattice Params: ', text)
            self.assertNotIn('Grid Shape:', text)
            # Output Tag is now a supported autoRSM key (it names the .nxs);
            # the non-custom-grid path tags it '<audit>_auto'.
            self.assertIn('Output Tag: ignored_auto', text)
            self.assertNotIn('U_S Record:', text)

    def test_autorsm_command_built_from_config(self):
        # The command is now built directly from configured paths -- no
        # intermediate command list -- as [python, autorsm, config].
        opts = {'python': '/correct/python', 'autorsm': '/correct/autoRSM.py'}
        config = '/logs/scan 42.txt'
        self.assertEqual(rsm_monitor.autorsm_command(opts, config),
                         ['/correct/python', '/correct/autoRSM.py', config])


if __name__ == '__main__':
    unittest.main()
