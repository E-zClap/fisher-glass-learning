from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.waterbirds_hidden_slice.common import (
    BlindSplit,
    load_blind_npz,
    require_current_scientific_implementation,
    scientific_implementation_manifest,
    stratified_partition_indices,
)


class BlindArtifactContractTests(unittest.TestCase):
    def _split(self) -> BlindSplit:
        return BlindSplit(
            row_id=np.arange(8, dtype=np.int64),
            h=np.arange(24, dtype=np.float32).reshape(8, 3),
            y=np.array([0, 1] * 4, dtype=np.int64),
            b_vl=np.linspace(-1.0, 1.0, 8, dtype=np.float32),
        )

    def test_round_trip_has_exact_blind_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blind.npz"
            self._split().save(path)
            loaded = load_blind_npz(path)
            self.assertEqual(set(loaded.arrays()), {"row_id", "h", "y", "b_vl"})
            np.testing.assert_array_equal(loaded.row_id, np.arange(8))

    def test_loader_rejects_metadata_like_key_before_exposure(self) -> None:
        split = self._split()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leaky.npz"
            np.savez_compressed(path, **split.arrays(), background=np.zeros(8, dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "metadata-like"):
                load_blind_npz(path)

    def test_label_only_partition_is_disjoint_complete_and_stratified(self) -> None:
        labels = np.array([0] * 20 + [1] * 20, dtype=np.int64)
        parts = stratified_partition_indices(labels, {"fit": 0.8, "cal": 0.2}, seed=17)
        joined = np.concatenate(list(parts.values()))
        np.testing.assert_array_equal(np.sort(joined), np.arange(labels.size))
        self.assertEqual(np.intersect1d(parts["fit"], parts["cal"]).size, 0)
        self.assertEqual(set(labels[parts["fit"]]), {0, 1})
        self.assertEqual(set(labels[parts["cal"]]), {0, 1})

    def test_implementation_manifest_changes_with_scientific_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gates.py"
            source.write_text("VERSION = 1\n", encoding="utf-8")
            first = scientific_implementation_manifest(root)
            source.write_text("VERSION = 2\n", encoding="utf-8")
            second = scientific_implementation_manifest(root)

        self.assertNotEqual(first["combined_sha256"], second["combined_sha256"])
        self.assertEqual(first["algorithm_version"], second["algorithm_version"])

    def test_downstream_rejects_legacy_or_stale_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "discover.py").write_text("VERSION = 1\n", encoding="utf-8")
            current = scientific_implementation_manifest(root)
            discovery = {"protocol": {"implementation": current}}
            require_current_scientific_implementation(
                discovery,
                label="test discovery",
                directory=root,
            )

            with self.assertRaisesRegex(RuntimeError, "no scientific implementation manifest"):
                require_current_scientific_implementation(
                    {"protocol": {}},
                    label="legacy discovery",
                    directory=root,
                )

            (root / "discover.py").write_text("VERSION = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "different scientific source code"):
                require_current_scientific_implementation(
                    discovery,
                    label="stale discovery",
                    directory=root,
                )

    def test_implementation_manifest_is_line_ending_portable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "discover.py"
            source.write_bytes(b"VALUE = 1\r\nOTHER = 2\r\n")
            windows = scientific_implementation_manifest(root)
            source.write_bytes(b"VALUE = 1\nOTHER = 2\n")
            linux = scientific_implementation_manifest(root)
        self.assertEqual(windows["combined_sha256"], linux["combined_sha256"])


if __name__ == "__main__":
    unittest.main()
