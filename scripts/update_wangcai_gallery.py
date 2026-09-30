#!/usr/bin/env python3
"""Index local Wangcai photos using structural/header checks, not full decoding.

Run without arguments for this skill, or --root PATH for another skill root.
Edit tags, caption and taken_at in the JSON; they and added_at survive renames
by SHA256. Missing files disappear from the current index. Concurrent writers
are not supported. No network requests or third-party packages are used.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tempfile
import zlib

PHOTO_DIR = Path('assets/wangcai/photos')
INDEX = Path('references/wangcai-gallery.json')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def image_size(data, suffix):
    """Check container boundaries and dimension headers; do not decode pixels."""
    if suffix == '.png':
        if data[:8] != b'\x89PNG\r\n\x1a\n':
            raise ValueError('invalid PNG signature')
        pos, size, seen_idat = 8, None, False
        while pos + 12 <= len(data):
            length = int.from_bytes(data[pos:pos + 4], 'big')
            kind = data[pos + 4:pos + 8]
            end = pos + 12 + length
            if end > len(data):
                raise ValueError('truncated PNG chunk')
            payload = data[pos + 8:end - 4]
            if zlib.crc32(kind + payload) != int.from_bytes(data[end - 4:end], 'big'):
                raise ValueError('PNG checksum mismatch')
            if pos == 8 and (kind != b'IHDR' or length != 13):
                raise ValueError('missing PNG IHDR')
            if kind == b'IHDR':
                if size is not None or length != 13:
                    raise ValueError('invalid PNG IHDR')
                size = struct.unpack('>II', payload[:8])
            if kind == b'IDAT':
                seen_idat = True
            if kind == b'IEND':
                if length or end != len(data) or not seen_idat:
                    raise ValueError('invalid PNG end')
                break
            pos = end
        else:
            raise ValueError('missing PNG IEND')
    elif suffix in {'.jpg', '.jpeg'}:
        if not data.startswith(b'\xff\xd8') or not data.endswith(b'\xff\xd9'):
            raise ValueError('invalid JPEG start/end')
        pos, size = 2, None
        while pos < len(data) - 2:
            if data[pos] != 255:
                raise ValueError('invalid JPEG marker')
            while pos < len(data) and data[pos] == 255:
                pos += 1
            if pos >= len(data):
                raise ValueError('truncated JPEG marker')
            marker = data[pos]
            pos += 1
            if pos + 2 > len(data):
                raise ValueError('truncated JPEG segment')
            length = int.from_bytes(data[pos:pos + 2], 'big')
            if length < 2 or pos + length > len(data) - 2:
                raise ValueError('invalid JPEG segment length')
            payload = data[pos + 2:pos + length]
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                if len(payload) < 6 or len(payload) != 6 + 3 * payload[5]:
                    raise ValueError('invalid JPEG frame header')
                height, width = struct.unpack('>HH', payload[1:5])
                size = width, height
            if marker == 0xDA:
                if len(payload) < 6 or len(payload) != 4 + 2 * payload[0] or size is None:
                    raise ValueError('invalid JPEG scan header')
                break
            pos += length
        else:
            raise ValueError('missing JPEG scan')
    elif suffix == '.webp':
        if len(data) < 20 or data[:4] != b'RIFF' or data[8:12] != b'WEBP' or int.from_bytes(data[4:8], 'little') + 8 != len(data):
            raise ValueError('invalid WebP RIFF container')
        pos, size, seen_image = 12, None, False
        while pos + 8 <= len(data):
            kind = data[pos:pos + 4]
            length = int.from_bytes(data[pos + 4:pos + 8], 'little')
            end = pos + 8 + length
            if end + (length % 2) > len(data):
                raise ValueError('truncated WebP chunk')
            payload = data[pos + 8:end]
            if kind == b'VP8L':
                if length < 5 or payload[0] != 0x2F or payload[4] >> 5:
                    raise ValueError('invalid WebP lossless header')
                bits = int.from_bytes(payload[1:5], 'little')
                frame_size = (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
                size = size or frame_size
                seen_image = True
            elif kind == b'VP8 ':
                if length < 10 or payload[3:6] != b'\x9d\x01\x2a' or payload[0] & 1:
                    raise ValueError('invalid WebP VP8 frame header')
                frame_size = tuple(v & 0x3FFF for v in struct.unpack('<HH', payload[6:10]))
                size = size or frame_size
                seen_image = True
            elif kind == b'VP8X':
                if length != 10:
                    raise ValueError('invalid WebP extended header')
                size = int.from_bytes(payload[4:7], 'little') + 1, int.from_bytes(payload[7:10], 'little') + 1
            pos = end + (length % 2)
        if pos != len(data) or not seen_image:
            raise ValueError('missing WebP image or incomplete chunk')
    else:
        raise ValueError('unsupported image extension')
    if size is None or min(size) <= 0:
        raise ValueError('missing or nonpositive image dimensions')
    return size


def load_index(path):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or data.get('schema_version', data.get('version')) != 1 or not isinstance(data.get('photos'), list):
            raise ValueError('invalid index schema')
        by_hash = {}
        for entry in data['photos']:
            if not isinstance(entry, dict) or not re.fullmatch(r'[0-9a-f]{64}', entry.get('sha256', '')):
                raise ValueError('invalid image hash')
            path_value = entry.get('path')
            if not isinstance(path_value, str):
                raise ValueError('invalid image path')
            relative = PurePosixPath(path_value)
            if relative.is_absolute() or '..' in relative.parts or '\\' in path_value or ':' in path_value or relative.parts[:3] != ('assets', 'wangcai', 'photos'):
                raise ValueError('index path outside photo directory')
            if any(type(entry.get(key)) is not int or entry[key] <= 0 for key in ('width', 'height')):
                raise ValueError('invalid indexed dimensions')
            if not isinstance(entry.get('tags'), list) or not all(isinstance(tag, str) for tag in entry['tags']):
                raise ValueError('invalid tags')
            if not isinstance(entry.get('caption'), str) or not isinstance(entry.get('added_at'), str) or not entry['added_at']:
                raise ValueError('invalid caption or added_at')
            if entry.get('taken_at') is not None and not isinstance(entry['taken_at'], str):
                raise ValueError('invalid taken_at')
            if entry['sha256'] in by_hash:
                raise ValueError('duplicate hash in index')
            by_hash[entry['sha256']] = entry
        return by_hash
    except (ValueError, TypeError) as error:
        raise ValueError(f'Existing gallery index is corrupt; left unchanged: {error}') from error


def update_gallery(root):
    root = Path(root).resolve()
    photo_dir, index = root / PHOTO_DIR, root / INDEX
    check_local_path(root, photo_dir)
    check_local_path(root, index)
    previous = load_index(index)
    if not stat.S_ISDIR(photo_dir.stat().st_mode):
        raise ValueError(f'Photo directory does not exist: {photo_dir}')
    report = {'added': 0, 'duplicates': 0, 'invalid': 0, 'invalid_files': [], 'duplicate_files': []}
    photos, hashes = [], set()
    now = datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
    candidates = []
    def raise_walk_error(error):
        raise error
    for directory, dirs, files in os.walk(photo_dir, onerror=raise_walk_error, followlinks=False):
        for name in dirs:
            check_local_path(root, Path(directory) / name)
        candidates.extend(Path(directory) / name for name in files if Path(name).suffix.lower() in EXTENSIONS)
    candidates.sort(key=lambda p: (p.relative_to(root).as_posix().casefold(), p.relative_to(root).as_posix()))
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        try:
            check_local_path(root, path)
            if not path.resolve().is_relative_to(photo_dir.resolve()) or not stat.S_ISREG(path.stat().st_mode):
                raise ValueError('not a regular local photo file')
            data = path.read_bytes()
            width, height = image_size(data, path.suffix.lower())
        except ValueError as error:
            report['invalid'] += 1
            report['invalid_files'].append({'path': relative, 'reason': str(error)})
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest in hashes:
            report['duplicates'] += 1
            report['duplicate_files'].append(relative)
            continue
        hashes.add(digest)
        old = previous.get(digest)
        entry = {'id': digest, 'path': relative, 'sha256': digest, 'width': width, 'height': height, 'tags': [], 'caption': '', 'taken_at': None, 'added_at': now}
        if old is not None:
            for key in ('tags', 'caption', 'taken_at', 'added_at'):
                entry[key] = old.get(key)
        else:
            report['added'] += 1
        photos.append(entry)
    document = {'schema_version': 1, 'validation': 'structural_headers_only_not_full_decode', 'updated_at': now, 'photos': photos}
    index.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=index.parent, prefix=index.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(document, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, index)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    report['total'] = len(photos)
    return report


def check_local_path(root, path):
    """Reject links/reparse points and escapes before reading or writing paths."""
    current = root
    for component in path.relative_to(root).parts:
        current /= component
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400):
            raise ValueError(f'Unsafe linked path; index unchanged: {current}')
    if not path.resolve().is_relative_to(root):
        raise ValueError(f'Path escapes skill root; index unchanged: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        report = update_gallery(args.root)
    except (ValueError, OSError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report['invalid'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
