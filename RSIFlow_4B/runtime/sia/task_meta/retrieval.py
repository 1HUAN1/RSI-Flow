"""Frozen offline FTS retrieval from public passage fields, never answer labels."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from collections.abc import Iterable, Iterator
from pathlib import Path

from .data import TaskRecord, sha256_file


def context_passages(payload: dict) -> Iterator[tuple[str, str]]:
    """Parse Hotpot dict-of-lists and 2Wiki JSON-string context formats."""
    context = payload.get("context", [])
    if isinstance(context, str):
        context = json.loads(context)
    if isinstance(context, dict):
        titles, sentences = context.get("title", []), context.get("sentences", [])
        if len(titles) != len(sentences):
            raise ValueError("Invalid Hotpot context alignment")
        context = zip(titles, sentences, strict=True)
    for title, sentences in context:
        if not isinstance(title, str) or not isinstance(sentences, list) or any(not isinstance(s, str) for s in sentences):
            raise ValueError("Corpus context must contain title and list of sentence strings")
        body = " ".join(sentences)
        if body.strip():
            yield title, body


def build_search_index(tasks: Iterable[TaskRecord], output_path: str | Path, *,
                       data_manifest_hash: str, external_corpus: str | Path | None = None) -> Path:
    """Build a fixed open-retrieval pool from evolve_train context passages.

    NQ is served by the same index; without an external Wikipedia corpus its
    coverage is limited and is explicitly recorded, never fabricated. Task
    questions, answers, supporting_facts and evidences are not indexed.
    """
    target = Path(output_path)
    if target.exists():
        raise FileExistsError("Frozen search index already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.with_suffix(target.suffix + ".building")
    if building.exists():
        raise FileExistsError("Incomplete corpus build exists")
    conn = sqlite3.connect(building)
    conn.executescript("""CREATE TABLE documents (doc_id TEXT PRIMARY KEY,title TEXT,body TEXT,source TEXT);
        CREATE VIRTUAL TABLE passages USING fts5(doc_id UNINDEXED,title,body,tokenize='unicode61');
        CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT);""")
    counts: dict[str, int] = {}

    def insert(title: str, body: str, source: str):
        doc_id = hashlib.sha256((title + "\n" + body).encode()).hexdigest()
        changed = conn.execute("INSERT OR IGNORE INTO documents VALUES (?,?,?,?)", (doc_id, title, body, source)).rowcount
        if changed:
            conn.execute("INSERT INTO passages VALUES (?,?,?)", (doc_id, title, body))
            counts[source] = counts.get(source, 0) + 1

    try:
        for i, task in enumerate(tasks):
            if task.domain != "searchqa" or task.split != "evolve_train":
                raise ValueError("Search corpus accepts only evolve_train public contexts")
            for title, body in context_passages(task.payload):
                insert(title, body, task.source)
            if i % 1000 == 0:
                conn.commit()
        corpus_hash = None
        if external_corpus:
            corpus_path = Path(external_corpus)
            corpus_hash = sha256_file(corpus_path)
            with corpus_path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if set(row) - {"doc_id", "title", "text", "source", "url"}:
                        raise ValueError("External corpus has undeclared fields; labels cannot be indexed")
                    if not isinstance(row.get("title"), str) or not isinstance(row.get("text"), str):
                        raise ValueError("External corpus requires string title/text")
                    insert(row["title"], row["text"], "external_corpus")
        metadata = {"schema_version": 1, "protocol": "open_retrieval", "retriever": "sqlite_fts5_bm25",
                    "sqlite_version": sqlite3.sqlite_version, "tokenizer": "unicode61", "sources": counts,
                    "data_manifest_hash": data_manifest_hash, "external_corpus_sha256": corpus_hash,
                    "nq_coverage": "external_corpus_unmeasured" if external_corpus else "training_context_pool_only_unmeasured",
                    "label_fields_indexed": False}
        conn.execute("INSERT INTO metadata VALUES ('manifest',?)", (json.dumps(metadata, sort_keys=True),))
        conn.execute("INSERT INTO passages(passages) VALUES ('optimize')")
        conn.commit()
        conn.close()
        building.rename(target)
        (target.with_suffix(target.suffix + ".manifest.json")).write_text(json.dumps({**metadata, "sha256": sha256_file(target)}, indent=2) + "\n", encoding="utf-8")
        return target
    except BaseException:
        conn.close()
        raise


class FrozenSearchIndex:
    def __init__(self, path: str | Path, *, expected_sha256: str | None = None, max_results: int = 5, max_chars: int = 4000):
        self.path = Path(path).resolve()
        if max_results < 1 or max_chars < 1:
            raise ValueError("Retrieval bounds must be positive")
        self.sha256 = sha256_file(self.path)
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError("Search index hash differs from frozen protocol")
        with closing(self._connect()) as conn:
            self.manifest = json.loads(conn.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        self.max_results, self.max_chars = max_results, max_chars

    def _connect(self):
        # HarnessForge executes tools on worker threads and rollout forks processes.
        # Open in the calling thread; never inherit/share a live SQLite connection.
        conn = sqlite3.connect(self.path.as_uri() + "?mode=ro&immutable=1", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def close(self):
        """Connections are scoped to each query and are already closed."""

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        if not isinstance(query, str) or len(query) > 4000:
            raise ValueError("Search query must be a string of at most 4000 characters")
        tokens = re.findall(r"\w+", query, flags=re.UNICODE)[:64]
        if not tokens:
            return []
        if top_k is not None and (not isinstance(top_k, int) or not 1 <= top_k <= self.max_results):
            raise ValueError("top_k exceeds declared retrieval budget")
        # Bind query and quote tokens; raw FTS operators are not an escape hatch.
        expression = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
        with closing(self._connect()) as conn:
            rows = conn.execute("""SELECT p.doc_id,p.title,p.body,bm25(passages) rank,d.source
                FROM passages p JOIN documents d USING(doc_id) WHERE passages MATCH ?
                ORDER BY rank,p.doc_id LIMIT ?""", (expression, top_k or self.max_results)).fetchall()
        return [{"doc_id": r["doc_id"], "title": r["title"], "text": r["body"][:self.max_chars],
                 "source": r["source"], "score": r["rank"], "truncated": len(r["body"]) > self.max_chars,
                 "corpus_hash": self.sha256} for r in rows]
