import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.prepare_data import digest, install, relocate, safe_relative, verify


class ProcessedDataReleaseTest(unittest.TestCase):
    def test_relocation_handles_dictionary_keys_without_changing_numbers(self):
        source = {'source_hashes': {'/old/input.tsv': 'hash'}, 'value': 0.2}
        self.assertEqual(relocate(source, '/old', '/new'),
                         {'source_hashes': {'/new/input.tsv': 'hash'}, 'value': 0.2})

    def test_rejects_escaping_paths(self):
        for value in ('../secret', '/absolute', 'dataset/../../escape', 'dataset\\escape'):
            with self.assertRaises(ValueError):
                safe_relative(value)

    def test_install_verifies_payload_and_detects_later_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            archive = root / 'data/inputs.tar.gz'
            content = b'c0\tp0\t1\n'
            with tarfile.open(archive, 'w:gz') as tar:
                info = tarfile.TarInfo('dataset/example.tsv')
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
            import hashlib
            manifest = dict(archive='data/inputs.tar.gz', archive_sha256=digest(archive),
                            original_root='/old', relocate_json=[],
                            files=[dict(path='dataset/example.tsv', bytes=len(content),
                                        sha256=hashlib.sha256(content).hexdigest())])
            install(root, manifest)
            verify(root, manifest)
            with self.assertRaises(FileExistsError):
                install(root, manifest)
            (root / 'dataset/example.tsv').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Installed input changed'):
                verify(root, manifest)

    def test_bad_hash_leaves_destination_unmodified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            archive = root / 'data/inputs.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                info = tarfile.TarInfo('dataset/example.tsv')
                info.size = 1
                tar.addfile(info, io.BytesIO(b'x'))
            manifest = dict(archive='data/inputs.tar.gz', archive_sha256=digest(archive),
                            files=[dict(path='dataset/example.tsv', bytes=1, sha256='bad')])
            with self.assertRaisesRegex(ValueError, 'Input checksum mismatch'):
                install(root, manifest)
            self.assertFalse((root / 'dataset').exists())


if __name__ == '__main__':
    unittest.main()
