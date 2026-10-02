"""Small generated audio fixtures test audit logic; none enters the dataset."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


SPEC = importlib.util.spec_from_file_location("inventory_gtzan", Path(__file__).resolve().parents[1] / "scripts" / "inventory_gtzan.py")
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class InventoryTests(unittest.TestCase):
    def test_native_decoding_statistics_include_every_frame_and_channel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "jazz" / "jazz.00000.wav"
            path.parent.mkdir()
            samples = np.tile(np.array([[0.25, -0.5], [0.0, 0.125]], dtype=np.float32), (40000, 1))
            sf.write(path, samples, 22050, subtype="FLOAT")
            row, preview = inventory.inspect_audio(path, root)
            self.assertEqual(row["recording_id"], "jazz.00000")
            self.assertEqual(row["decode_status"], "ok")
            self.assertEqual(row["decoded_frames"], 80000)
            self.assertEqual(row["sample_count"], 160000)
            self.assertEqual(row["finite_sample_count"], 160000)
            self.assertAlmostEqual(row["rms"], float(np.sqrt(np.mean(samples.astype(float) ** 2))))
            self.assertEqual(row["peak"], 0.5)
            self.assertEqual(row["zero_fraction"], 0.25)
            self.assertEqual(preview.shape, (1102, 2))
            self.assertIn("unexpected_channels", row["anomalies"])

    def test_failed_decode_is_retained_and_missing_ids_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            path = root / "classical" / "classical.00000.wav"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"not audio")
            (root / "README.txt").write_text("fixture")
            output = Path(directory) / "metadata"
            summary = inventory.run_inventory(root, output)
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["total_audio_files"], 1)
            self.assertEqual(summary["decode_status_counts"], {"failed": 1})
            self.assertEqual(len(summary["by_genre"]["classical"]["missing_expected_ids"]), 99)
            self.assertEqual(summary["ignored_non_audio_files"], ["README.txt"])
            self.assertIn("classical/classical.00000.wav", (output / "gtzan_inventory.csv").read_text())
            self.assertEqual(json.loads((output / "gtzan_summary.json").read_text())["status"], "failed")
            self.assertTrue((output / "gtzan_waveforms.svg").exists())

    def test_genre_mismatch_cannot_be_mistaken_for_correct_expected_id(self):
        genre, recording_id, anomalies = inventory.identify(Path("raw/jazz/metal.00000.wav"), Path("raw"))
        self.assertEqual(genre, "jazz")
        self.assertEqual(recording_id, "jazz.metal.00000")
        self.assertIn("genre_filename_mismatch", anomalies)

    def test_appledouble_classification_requires_both_name_and_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            genre = root / "jazz"
            genre.mkdir(parents=True)
            real_failure = genre / "jazz.00054.wav"
            real_failure.write_bytes(b"corrupt real audio")
            verified = genre / "._jazz.00054.wav"
            verified.write_bytes(inventory.APPLEDOUBLE_MAGIC + b"metadata")
            wrong_magic = genre / "._jazz.00055.wav"
            wrong_magic.write_bytes(b"not verified metadata")
            wrong_name = genre / "jazz.00056.wav"
            wrong_name.write_bytes(inventory.APPLEDOUBLE_MAGIC + b"metadata")
            output = Path(directory) / "metadata"
            summary = inventory.run_inventory(root, output)
            self.assertEqual(summary["total_discovered_files"], 4)
            self.assertEqual(summary["total_audio_files"], 3)
            self.assertEqual(summary["decode_status_counts"], {"failed": 3})
            self.assertEqual(summary["appledouble_metadata_count"], 1)
            self.assertEqual(summary["appledouble_metadata_files"][0]["relative_path"], "jazz/._jazz.00054.wav")
            self.assertEqual(summary["appledouble_metadata_files"][0]["header_magic_hex"], "00051607")
            self.assertEqual(summary["unverified_appledouble_candidates"][0]["relative_path"], "jazz/._jazz.00055.wav")
            self.assertTrue(any("unverified AppleDouble" in failure for failure in summary["hard_failures"]))
            self.assertTrue(any("jazz/jazz.00054.wav: decode_status=failed" == failure for failure in summary["hard_failures"]))
            self.assertIn("jazz/jazz.00054.wav", (output / "gtzan_inventory.csv").read_text())
            self.assertNotIn("jazz/._jazz.00054.wav", (output / "gtzan_inventory.csv").read_text())
            self.assertEqual(verified.read_bytes(), inventory.APPLEDOUBLE_MAGIC + b"metadata")

    def test_nonfinite_samples_are_explicit_hard_anomalies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "pop.00000.wav"
            sf.write(path, np.array([0.0, np.nan, np.inf], dtype=np.float32), 22050, subtype="FLOAT")
            row, _ = inventory.inspect_audio(path, root)
            self.assertEqual(row["nonfinite_sample_count"], 2)
            self.assertIn("nonfinite_samples", row["anomalies"])
            self.assertIn("nonfinite_samples", inventory.HARD_ANOMALIES)

    def test_metadata_cannot_be_written_inside_raw_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                inventory.run_inventory(root, root / "metadata")


if __name__ == "__main__":
    unittest.main()
