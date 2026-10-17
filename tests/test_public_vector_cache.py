import hashlib
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest

from scripts.public_vector_cache import PublicVectorCache


class Backend:
    identity = 'fixture'
    model_hash = 'fixture-model'
    dimension = 2

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        return [[0., 1.] for _ in texts]


class PublicVectorCacheTests(unittest.TestCase):
    def test_public_cache_is_read_only_and_model_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.db'
            with sqlite3.connect(path) as con:
                con.execute('CREATE TABLE vectors(content_hash,model_hash,dimension,vector)')
                for text, model in [('cached', 'fixture-model'), ('other', 'wrong-model')]:
                    con.execute('INSERT INTO vectors VALUES(?,?,?,?)', (
                        hashlib.sha256(text.encode()).hexdigest(), model, 2,
                        struct.pack('<2f', 1., 0.)))
            before = path.read_bytes()
            backend = Backend()
            cache = PublicVectorCache(backend, path)
            self.assertEqual(cache.embed(['cached', 'missing', 'other']),
                             [(1., 0.), [0., 1.], [0., 1.]])
            self.assertEqual(backend.calls, [['missing', 'other']])
            self.assertEqual((cache.hits, cache.misses), (1, 2))
            self.assertEqual(before, path.read_bytes())
            self.assertEqual((cache.identity, cache.model_hash), ('fixture', 'fixture-model'))

    def test_missing_path_does_not_create_a_database(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missing.db'
            with self.assertRaises(FileNotFoundError):
                PublicVectorCache(Backend(), path)
            self.assertFalse(path.exists())

    def test_corrupt_vector_fails_instead_of_silently_using_it(self):
        for blob in (b'', struct.pack('<2f', float('nan'), 0.)):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'cache.db'
                with sqlite3.connect(path) as con:
                    con.execute('CREATE TABLE vectors(content_hash,model_hash,dimension,vector)')
                    con.execute('INSERT INTO vectors VALUES(?,?,?,?)', (
                        hashlib.sha256(b'cached').hexdigest(), 'fixture-model', 2, blob))
                backend = Backend()
                with self.assertRaises(ValueError):
                    PublicVectorCache(backend, path).embed(['cached'])
                self.assertEqual(backend.calls, [])
