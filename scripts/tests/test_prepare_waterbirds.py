from __future__ import annotations

import csv
import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "prepare_waterbirds.py"
TINY_COUNTS = {(0, 0): 1, (1, 1): 1, (2, 2): 1}


class PrepareWaterbirdsTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT.exists(), "The portable data preparation command is missing")
        spec = importlib.util.spec_from_file_location("prepare_waterbirds", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def make_archive(self, path: Path, *, omit_image=False, wrong_split=False, extra=None):
        rows = []
        for index, (split, group) in enumerate(TINY_COUNTS):
            rows.append({"img_id": index, "img_filename": f"{index+1:03d}.Bird/image.jpg",
                         "y": group // 2, "place": group % 2,
                         "split": 0 if wrong_split else split, "place_filename": "background.jpg"})
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        with tarfile.open(path, "w:gz") as archive:
            records = [("waterbird_complete95_forest2water2/metadata.csv", text.getvalue().encode())]
            records += [("waterbird_complete95_forest2water2/" + row["img_filename"], b"image fixture")
                        for row in rows[:2 if omit_image else 3]]
            for name, content in records:
                member = tarfile.TarInfo(name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
            if extra is not None:
                archive.addfile(extra)

    def test_complete_archive_gets_validated_wilds_marker(self):
        module = self.load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "fixture.tar.gz"
            self.make_archive(archive)
            target = module.stage_archive(archive, root / "data", expected_counts=TINY_COUNTS, expected_species=3)
            self.assertEqual(target.name, "waterbirds_v1.0")
            self.assertTrue((target / "RELEASE_v1.0.txt").is_file())
            report = module.validate_dataset(target, expected_counts=TINY_COUNTS, expected_species=3)
            self.assertEqual(report["n_images"], 3)

    def test_missing_image_never_publishes_dataset_or_marker(self):
        module = self.load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "fixture.tar.gz"
            self.make_archive(archive, omit_image=True)
            with self.assertRaisesRegex(ValueError, "missing image"):
                module.stage_archive(archive, root / "data", expected_counts=TINY_COUNTS, expected_species=3)
            self.assertFalse((root / "data/waterbirds_v1.0").exists())

    def test_wrong_split_counts_never_publish_marker(self):
        module = self.load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "fixture.tar.gz"
            self.make_archive(archive, wrong_split=True)
            with self.assertRaisesRegex(ValueError, "split/group counts"):
                module.stage_archive(archive, root / "data", expected_counts=TINY_COUNTS, expected_species=3)
            self.assertFalse((root / "data/waterbirds_v1.0").exists())

    def test_parent_escape_and_symlinks_are_rejected(self):
        module = self.load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, kind in (("../escaped", tarfile.REGTYPE), ("linked", tarfile.SYMTYPE)):
                with self.subTest(name=name):
                    archive = root / "fixture.tar.gz"
                    extra = tarfile.TarInfo(name)
                    extra.type = kind
                    extra.linkname = "../../escaped"
                    self.make_archive(archive, extra=extra)
                    with self.assertRaisesRegex(ValueError, "unsafe archive"):
                        module.stage_archive(archive, root / "data", expected_counts=TINY_COUNTS, expected_species=3)
                    self.assertFalse((root / "escaped").exists())
                    self.assertFalse((root / "data/waterbirds_v1.0").exists())

    def test_existing_dataset_is_not_overwritten(self):
        module = self.load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "fixture.tar.gz"
            self.make_archive(archive)
            target = root / "data/waterbirds_v1.0"
            target.mkdir(parents=True)
            (target / "keep.txt").write_text("existing data")
            with self.assertRaises(FileExistsError):
                module.stage_archive(archive, root / "data", expected_counts=TINY_COUNTS, expected_species=3)
            self.assertEqual((target / "keep.txt").read_text(), "existing data")


if __name__ == "__main__":
    unittest.main()
