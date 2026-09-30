import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/update_wangcai_gallery.py'


def png(width=2, height=3):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b'\x00' * (height * (1 + width * 3)))) + chunk(b'IEND', b''))


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.exists(), 'Gallery indexer has not been implemented')
        spec = importlib.util.spec_from_file_location('gallery', SCRIPT)
        self.gallery = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.gallery)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.photos = self.root / 'assets/wangcai/photos'
        self.photos.mkdir(parents=True)
        self.index = self.root / 'references/wangcai-gallery.json'

    def run_update(self):
        return self.gallery.update_gallery(self.root)

    def read(self):
        return json.loads(self.index.read_text(encoding='utf-8'))

    def test_deduplicates_in_deterministic_filename_order(self):
        (self.photos / 'z.png').write_bytes(png())
        (self.photos / 'a.PNG').write_bytes(png())
        report = self.run_update()
        self.assertEqual((report['added'], report['duplicates'], report['invalid']), (1, 1, 0))
        entry = self.read()['photos'][0]
        self.assertEqual(entry['path'], 'assets/wangcai/photos/a.PNG')
        self.assertEqual((entry['width'], entry['height']), (2, 3))
        self.assertEqual(len(entry['sha256']), 64)
        self.assertEqual(entry.get('id'), entry['sha256'])
        self.assertEqual(self.read().get('schema_version'), 1)

    def test_preserves_metadata_by_hash_across_rename(self):
        old = self.photos / 'old.png'
        old.write_bytes(png())
        self.run_update()
        data = self.read()
        data['photos'][0].update(tags=['旺财'], caption='睡觉', taken_at='2026-08-27', added_at='2026-01-01T00:00:00Z')
        self.index.write_text(json.dumps(data), encoding='utf-8')
        old.rename(self.photos / 'new.png')
        self.assertEqual(self.run_update()['added'], 0)
        entry = self.read()['photos'][0]
        self.assertEqual(entry['tags'], ['旺财'])
        self.assertEqual(entry['caption'], '睡觉')
        self.assertEqual(entry['taken_at'], '2026-08-27')
        self.assertEqual(entry['added_at'], '2026-01-01T00:00:00Z')
        self.assertIsNotNone(entry.get('id'))
        self.assertEqual(entry.get('id'), data['photos'][0].get('id'))
        self.assertTrue(entry['path'].endswith('/new.png'))

    def test_rejects_bad_headers_truncation_and_zero_dimensions(self):
        for name, data in {'bad.jpg': b'garbage', 'zero.png': png(0), 'truncated.png': png()[:-8], 'wrong.jpg': png()}.items():
            (self.photos / name).write_bytes(data)
        report = self.run_update()
        self.assertEqual(report['invalid'], 4)
        self.assertEqual(self.read()['photos'], [])

    def test_jpeg_and_webp_structural_headers(self):
        # Synthetic containers test header parsing only, not full image decoding.
        jpeg = b'\xff\xd8\xff\xc0\x00\x0b\x08\x00\x03\x00\x02\x01\x01\x11\x00\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\x11\xff\xd9'
        payload = b'\x2f' + (1 | (2 << 14)).to_bytes(4, 'little') + b'\x00'
        webp = b'RIFF' + struct.pack('<I', 12 + len(payload)) + b'WEBPVP8L' + struct.pack('<I', len(payload)) + payload
        (self.photos / 'a.jpeg').write_bytes(jpeg)
        (self.photos / 'b.webp').write_bytes(webp)
        report = self.run_update()
        self.assertEqual((report['added'], report['invalid']), (2, 0))
        self.assertTrue(all((e['width'], e['height']) == (2, 3) for e in self.read()['photos']))

    def test_corrupt_existing_index_never_overwritten(self):
        self.index.parent.mkdir()
        for content in ['{broken', '{}', '{"version":1,"photos":[{}]}']:
            self.index.write_text(content, encoding='utf-8')
            with self.assertRaises(ValueError):
                self.run_update()
            self.assertEqual(self.index.read_text(encoding='utf-8'), content)

    def test_migrates_legacy_version_preserving_metadata(self):
        (self.photos / 'a.png').write_bytes(png())
        self.run_update()
        data = self.read()
        data.pop('schema_version', None)
        data['version'] = 1
        data['photos'][0].pop('id', None)
        data['photos'][0].update(tags=['original'], caption='kept', taken_at='2026-08-27', added_at='2026-01-01T00:00:00Z')
        self.index.write_text(json.dumps(data), encoding='utf-8')
        self.run_update()
        migrated = self.read()
        self.assertEqual(migrated.get('schema_version'), 1)
        self.assertNotIn('version', migrated)
        entry = migrated['photos'][0]
        self.assertEqual(entry['id'], entry['sha256'])
        for key in ('tags', 'caption', 'taken_at', 'added_at'):
            self.assertEqual(entry[key], data['photos'][0][key])

    def test_cli_invalid_image_returns_nonzero_and_report(self):
        (self.photos / 'bad.jpg').write_bytes(b'not an image')
        result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(self.root)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)['invalid'], 1)

    def test_webp_lossy_and_extended_with_real_image_chunk_headers(self):
        def chunk(kind, payload):
            return kind + struct.pack('<I', len(payload)) + payload + (b'\0' if len(payload) % 2 else b'')
        lossy = chunk(b'VP8 ', b'\x00\x00\x00\x9d\x01\x2a' + struct.pack('<HH', 2, 3))
        extended = chunk(b'VP8X', b'\0\0\0\0' + (1).to_bytes(3, 'little') + (2).to_bytes(3, 'little'))
        for name, chunks in [('lossy.webp', lossy), ('extended.webp', extended + lossy)]:
            (self.photos / name).write_bytes(b'RIFF' + struct.pack('<I', 4 + len(chunks)) + b'WEBP' + chunks)
        report = self.run_update()
        self.assertEqual((report['added'], report['invalid']), (2, 0))
        self.assertTrue(all((e['width'], e['height']) == (2, 3) for e in self.read()['photos']))

    def test_replace_failure_preserves_previous_index(self):
        (self.photos / 'a.png').write_bytes(png())
        self.run_update()
        original = self.index.read_bytes()
        (self.photos / 'b.png').write_bytes(png(4, 5))
        with patch.object(self.gallery.os, 'replace', side_effect=OSError('simulated failure')):
            with self.assertRaises(OSError):
                self.run_update()
        self.assertEqual(self.index.read_bytes(), original)
        self.assertEqual(list(self.index.parent.glob('*.tmp')), [])

    def test_read_permission_error_preserves_previous_index(self):
        photo = self.photos / 'a.png'
        photo.write_bytes(png())
        self.run_update()
        data = self.read()
        data['photos'][0]['caption'] = 'manual metadata must survive'
        self.index.write_text(json.dumps(data), encoding='utf-8')
        original = self.index.read_bytes()
        with patch.object(Path, 'read_bytes', side_effect=PermissionError('temporarily locked')):
            with self.assertRaises(OSError):
                self.run_update()
        self.assertEqual(self.index.read_bytes(), original)

    def test_external_directory_links_rejected_without_overwrite(self):
        for relative in ('assets', 'assets/wangcai', 'assets/wangcai/photos', 'references'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as sandbox:
                sandbox = Path(sandbox)
                root = sandbox / 'skill'
                outside = sandbox / 'outside'
                root.mkdir()
                outside.mkdir()
                link = root / relative
                link.parent.mkdir(parents=True, exist_ok=True)
                if os.name == 'nt':
                    result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(outside)], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    link.symlink_to(outside, target_is_directory=True)
                try:
                    photos = root / 'assets/wangcai/photos'
                    photos.mkdir(parents=True, exist_ok=True)
                    (photos / 'a.png').write_bytes(png())
                    index = root / 'references/wangcai-gallery.json'
                    index.parent.mkdir(parents=True, exist_ok=True)
                    original = b'{"version": 1, "photos": []}\n'
                    index.write_bytes(original)
                    with self.assertRaises(ValueError):
                        self.gallery.update_gallery(root)
                    self.assertEqual(index.read_bytes(), original)
                finally:
                    # Remove only the explicitly created link, never recurse through it.
                    if os.name == 'nt':
                        os.rmdir(link)
                    else:
                        link.unlink()


if __name__ == '__main__':
    unittest.main()
