"""Native HarnessForge tools must query the frozen corpus from worker threads."""
import concurrent.futures
import multiprocessing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from sia.task_meta.retrieval import FrozenSearchIndex


def query_in_child(index):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(index.search, ['director'] * 4))


class SearchThreads(unittest.TestCase):
    def test_thread_and_fork_queries_keep_exact_frozen_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'search.sqlite'
            with sqlite3.connect(path) as conn:
                conn.executescript('''
                    CREATE TABLE documents (doc_id TEXT PRIMARY KEY,title TEXT,body TEXT,source TEXT);
                    CREATE VIRTUAL TABLE passages USING fts5(doc_id UNINDEXED,title,body);
                    CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
                    INSERT INTO metadata VALUES ('manifest','{}');
                    INSERT INTO documents VALUES ('one','Example','The film director is Alice.','fixture');
                    INSERT INTO passages VALUES ('one','Example','The film director is Alice.');
                ''')
            index = FrozenSearchIndex(path)
            expected = index.search('director')
            self.assertEqual(expected[0]['text'], 'The film director is Alice.')
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                self.assertEqual(list(pool.map(index.search, ['director'] * 64)), [expected] * 64)
            with concurrent.futures.ProcessPoolExecutor(max_workers=2,
                    mp_context=multiprocessing.get_context('fork')) as pool:
                self.assertEqual(list(pool.map(query_in_child, [index] * 2)), [[expected] * 4] * 2)
            conn = index._connect()
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    conn.execute("DELETE FROM documents")
            finally:
                conn.close()
            index.close()
