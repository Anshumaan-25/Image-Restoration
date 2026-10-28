"""Archive integrity and non-overwriting extraction tests using tiny local ZIPs."""
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
import zipfile
from PIL import Image

from scripts.prepare_div2k import ArchiveSpec, extract_verified_archive, immutable_write, verify_members


class AcquisitionTests(unittest.TestCase):
    """Dataset acquisition must not replace user images or accept wrong archives."""

    def test_verified_extraction_and_existing_image_protection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "tiny.zip"
            image = BytesIO()
            Image.new("RGB", (12, 13), (10, 40, 100)).save(image, format="PNG")
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("tiny/0001.png", image.getvalue())
                output.writestr("tiny/0002.png", image.getvalue())
            spec = ArchiveSpec("tiny", 1, 2, archive.stat().st_size)
            report = extract_verified_archive(archive, root / "images", spec)
            self.assertEqual(report["zip_crc"], "passed")
            report = extract_verified_archive(archive, root / "images", spec)
            self.assertEqual(report["existing_images_reused"], 2)
            target = root / "images/tiny/0001.png"
            target.write_bytes(b"existing different file")
            with self.assertRaises(FileExistsError):
                extract_verified_archive(archive, root / "images", spec)
            self.assertEqual(target.read_bytes(), b"existing different file")

    def test_wrong_member_and_size_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../outside.png", b"invalid")
            spec = ArchiveSpec("tiny", 1, 1, archive.stat().st_size)
            with zipfile.ZipFile(archive) as source, self.assertRaises(ValueError):
                verify_members(source, spec)
            with self.assertRaises(ValueError):
                extract_verified_archive(archive, Path(temporary), ArchiveSpec("tiny", 1, 1, 1))

    def test_immutable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            immutable_write(path, "original\n")
            immutable_write(path, "original\n")
            with self.assertRaises(FileExistsError):
                immutable_write(path, "changed\n")
            self.assertEqual(path.read_text(), "original\n")
