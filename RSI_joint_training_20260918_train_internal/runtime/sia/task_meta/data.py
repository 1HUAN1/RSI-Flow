"""Streaming, immutable three-domain data manifests and resumable balanced windows.

Only trusted controllers open this index: source rows can contain verifier labels.
The Task receives ``TaskRecord.public_payload()`` and never source paths or labels.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

DOMAINS = ("tool_use", "code", "searchqa")


def canonical_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).lower()).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(value: str) -> str:
    return hashlib.sha256(canonical_text(value).encode()).hexdigest()


def jsonl_rows(path: Path) -> Iterator[tuple[int, int, dict]]:
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                return
            if line.strip():
                yield offset, len(line), json.loads(line)


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    domain: str
    source: str
    split: str
    prompt: str
    payload: dict
    source_hash: str = ""
    content_hash: str = ""

    def public_payload(self) -> dict:
        """Strict allowlist; tests, solutions, answers and checklists stay trusted."""
        public = {"task_id": self.task_id, "domain": self.domain, "prompt": self.prompt}
        if self.domain == "code":
            # Only independently declared public samples; never derive these from tests.
            public["public_tests"] = self.payload.get("public_tests", [])
        return public


def _prompt(row: dict, domain: str) -> str:
    key = {"tool_use": "task", "code": "problem", "searchqa": "question"}[domain]
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing/non-string {key} in {domain} record")
    return value


def _identity(row: dict, domain: str, source: str) -> tuple[str, str]:
    question_hash = content_hash(_prompt(row, domain))
    native_id = row.get("task_id", row.get("id", row.get("_id", question_hash)))
    if domain == "tool_use":
        env = row.get("canonical_env_id", row.get("env_id"))
        if env is None:
            raise ValueError("EnvScaler requires canonical environment ID")
        native_id = f"{env}:{native_id}"
    return f"{source}:{native_id}", question_hash


def _source_name(item: dict) -> str:
    name = item["dataset_name"].lower()
    if "envscaler" in name:
        return "envscaler"
    if "taco" in name:
        return "deepcoder_taco"
    if "nq" in name:
        return "nq_open"
    if "hotpot" in name:
        return "hotpotqa"
    if "2wiki" in name:
        return "2wiki"
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def build_manifest(
    source_manifest: str | Path,
    output_dir: str | Path,
    *,
    split_seed: int = 42,
    search_dev_fraction: float = 0.1,
    probe_per_domain: dict[str, int] | None = None,
) -> Path:
    """Build once, using bounded memory and preserving prepared Tool/Code splits.

    Search questions are grouped by normalized content across all three sources;
    duplicate source IDs with different contents are rejected. Final question
    overlap hashes are read only by this trusted preparation procedure.
    """
    if not 0 < search_dev_fraction < 1:
        raise ValueError("search_dev_fraction must be between zero and one")
    probe_per_domain = probe_per_domain or dict.fromkeys(DOMAINS, 64)
    if set(probe_per_domain) != set(DOMAINS) or any(n < 1 for n in probe_per_domain.values()):
        raise ValueError("probe_per_domain requires positive quotas for all three domains")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "tasks.sqlite"
    if target.exists():
        raise FileExistsError("Immutable data manifest already exists")
    staging = output / "tasks.sqlite.building"
    if staging.exists():
        raise FileExistsError("Incomplete preparation exists; inspect before removing it")
    catalog_path = Path(source_manifest)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    conn = sqlite3.connect(staging)
    conn.executescript("""
        CREATE TABLE sources (path TEXT PRIMARY KEY, hash TEXT, metadata TEXT);
        CREATE TABLE final_hashes (domain TEXT, hash TEXT, PRIMARY KEY(domain, hash));
        CREATE TABLE final_ids (task_id TEXT PRIMARY KEY);
        CREATE TABLE tasks (
          task_id TEXT PRIMARY KEY, domain TEXT, source TEXT, split TEXT,
          content_hash TEXT, group_key TEXT, path TEXT, offset INTEGER, length INTEGER,
          order_key TEXT, seq INTEGER
        );
        CREATE TABLE omissions (source TEXT, task_id TEXT, reason TEXT);
        CREATE TABLE probe (task_id TEXT PRIMARY KEY);
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
        CREATE INDEX task_group ON tasks(domain, content_hash);
    """)
    try:
        # Only hashes enter the audit; report labels and predictions never reach Meta.
        for item in catalog["datasets"]:
            domain = item["domain"]
            if item["role"] != "final_test" or domain != "searchqa":
                continue
            path = Path(item["local_path"])
            conn.execute("INSERT OR IGNORE INTO sources VALUES (?,?,?)", (str(path), sha256_file(path), json.dumps(item)))
            for _, _, row in jsonl_rows(path):
                conn.execute("INSERT OR IGNORE INTO final_hashes VALUES (?,?)", (domain, content_hash(row["question"])))
                final_id, _ = _identity(row, domain, _source_name(item))
                conn.execute("INSERT OR IGNORE INTO final_ids VALUES (?)", (final_id,))
        for item in catalog["datasets"]:
            if item["role"] not in {"train", "validation"}:
                continue
            if "environments" in item["dataset_name"].lower():
                path = Path(item["local_path"])
                conn.execute("INSERT OR IGNORE INTO sources VALUES (?,?,?)", (str(path), sha256_file(path), json.dumps(item)))
                continue
            domain = item["domain"]
            if domain not in DOMAINS:
                raise ValueError(f"Unapproved training domain: {domain}")
            path = Path(item["local_path"])
            source = _source_name(item)
            conn.execute("INSERT OR IGNORE INTO sources VALUES (?,?,?)", (str(path), sha256_file(path), json.dumps(item)))
            for offset, length, row in jsonl_rows(path):
                task_id, qhash = _identity(row, domain, source)
                split = "evolve_train" if item["role"] == "train" else "search_dev"
                group_key = str(row.get("canonical_env_id", row.get("env_id"))) if domain == "tool_use" else qhash
                if domain == "searchqa":
                    if item["role"] != "train":
                        raise ValueError("Search dev must be isolated from training sources, never official dev")
                    if conn.execute("SELECT 1 FROM final_hashes WHERE domain=? AND hash=?", (domain, qhash)).fetchone():
                        conn.execute("INSERT INTO omissions VALUES (?,?,?)", (source, task_id, "final_question_overlap"))
                        continue
                    if conn.execute("SELECT 1 FROM final_ids WHERE task_id=?", (task_id,)).fetchone():
                        conn.execute("INSERT INTO omissions VALUES (?,?,?)", (source, task_id, "final_task_id_overlap"))
                        continue
                    draw = int(hashlib.sha256(f"{split_seed}:search-split:{group_key}".encode()).hexdigest(), 16) / 2**256
                    split = "search_dev" if draw < search_dev_fraction else "evolve_train"
                duplicate = conn.execute("SELECT content_hash, split FROM tasks WHERE task_id=?", (task_id,)).fetchone()
                if duplicate:
                    if duplicate != (qhash, split):
                        raise ValueError(f"Conflicting task ID or split: {task_id}")
                    conn.execute("INSERT INTO omissions VALUES (?,?,?)", (source, task_id, "duplicate_task_id"))
                    continue
                # One source-independent question per Search pool; avoids duplicate supervision.
                if domain == "searchqa" and conn.execute("SELECT 1 FROM tasks WHERE domain=? AND content_hash=?", (domain, qhash)).fetchone():
                    conn.execute("INSERT INTO omissions VALUES (?,?,?)", (source, task_id, "duplicate_question_content"))
                    continue
                order_key = hashlib.sha256(f"{split_seed}:order:{task_id}".encode()).hexdigest()
                conn.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,NULL)", (task_id, domain, source, split, qhash, group_key, str(path), offset, length, order_key))
            conn.commit()
        overlaps = conn.execute("SELECT domain,group_key FROM tasks GROUP BY domain,group_key HAVING COUNT(DISTINCT split)>1 LIMIT 1").fetchone()
        if overlaps:
            raise ValueError(f"Prepared split leakage; refusing to resplit silently: {overlaps}")
        conn.execute("CREATE INDEX task_split ON tasks(domain, split, order_key)")
        for domain in DOMAINS:
            # Window order is fixed at preparation, independent of generation/resume.
            conn.execute("""WITH ordering AS (SELECT task_id, ROW_NUMBER() OVER (ORDER BY order_key, task_id)-1 AS n
                FROM tasks WHERE domain=? AND split='evolve_train')
                UPDATE tasks SET seq=(SELECT n FROM ordering WHERE ordering.task_id=tasks.task_id)
                WHERE domain=? AND split='evolve_train'""", (domain, domain))
            ids = conn.execute("SELECT task_id FROM tasks WHERE domain=? AND split='search_dev' ORDER BY order_key,task_id LIMIT ?", (domain, probe_per_domain[domain])).fetchall()
            if len(ids) < probe_per_domain[domain]:
                raise ValueError(f"{domain} dev pool has {len(ids)}, below requested fixed probe {probe_per_domain[domain]}")
            conn.executemany("INSERT INTO probe VALUES (?)", ids)
        conn.execute("CREATE INDEX task_window ON tasks(domain, split, seq)")
        settings = {"schema_version": 1, "split_seed": split_seed, "search_dev_fraction": search_dev_fraction,
                    "probe_per_domain": probe_per_domain, "source_manifest_hash": sha256_file(catalog_path),
                    "search_protocol": "open_retrieval_frozen_training_context_corpus",
                    "window_policy": "explicit_domain_quotas_without_replacement; exhausted_domains_return_zero; continue_remaining_domains",
                    "normalization": "NFKC lowercase collapse-whitespace"}
        conn.execute("INSERT INTO metadata VALUES ('settings',?)", (json.dumps(settings, sort_keys=True),))
        conn.commit()
        conn.close()
        staging.rename(target)
        store = ManifestStore(target)
        summary = {**settings, "manifest_sha256": sha256_file(target), "counts": store.counts(),
                   "omissions": store.omissions(), "probe_sha256": content_hash(json.dumps([t.task_id for t in store.probe()])),
                   "status": "PREPARED_NOT_TRAINED", "full_coverage": False}
        (output / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        store.close()
        return target
    except BaseException:
        conn.close()
        raise


class ManifestStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.conn = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def validate_sources(self) -> None:
        for source in self.conn.execute("SELECT path,hash FROM sources"):
            if sha256_file(Path(source["path"])) != source["hash"]:
                raise ValueError(f"Frozen data source changed: {source['path']}")

    def _record(self, row: sqlite3.Row) -> TaskRecord:
        with Path(row["path"]).open("rb") as stream:
            stream.seek(row["offset"])
            payload = json.loads(stream.read(row["length"]))
        if content_hash(_prompt(payload, row["domain"])) != row["content_hash"]:
            raise ValueError("Source content differs from immutable manifest")
        source_hash = self.conn.execute("SELECT hash FROM sources WHERE path=?", (row["path"],)).fetchone()[0]
        return TaskRecord(row["task_id"], row["domain"], row["source"], row["split"], _prompt(payload, row["domain"]), payload, source_hash, row["content_hash"])

    def window(self, cursor: dict[str, int] | None, quotas: dict[str, int]) -> tuple[list[TaskRecord], dict[str, int]]:
        if set(quotas) != set(DOMAINS) or any(n < 1 for n in quotas.values()):
            raise ValueError("Window requires positive quotas for all three domains")
        following = dict.fromkeys(DOMAINS, 0) | (cursor or {})
        if set(following) != set(DOMAINS) or any(not isinstance(n, int) or n < 0 for n in following.values()):
            raise ValueError("Invalid coverage cursor")
        records = []
        for domain in DOMAINS:
            rows = self.conn.execute("SELECT * FROM tasks WHERE domain=? AND split='evolve_train' AND seq>=? ORDER BY seq LIMIT ?", (domain, following[domain], quotas[domain])).fetchall()
            records.extend(self._record(row) for row in rows)
            following[domain] += len(rows)
        return records, following

    def probe(self) -> list[TaskRecord]:
        return [self._record(row) for row in self.conn.execute("SELECT t.* FROM tasks t JOIN probe p USING(task_id) ORDER BY domain, order_key, task_id")]

    def iter_split(self, split: str, domain: str | None = None) -> Iterator[TaskRecord]:
        if split not in {"evolve_train", "search_dev"}:
            raise ValueError("Report evaluation is separate from evolution data")
        for row in self.conn.execute("SELECT * FROM tasks WHERE split=? AND (? IS NULL OR domain=?) ORDER BY domain,order_key", (split, domain, domain)):
            yield self._record(row)

    def counts(self) -> dict:
        result = {domain: {"evolve_train": 0, "search_dev": 0, "probe": 0} for domain in DOMAINS}
        for row in self.conn.execute("SELECT domain,split,COUNT(*) n FROM tasks GROUP BY domain,split"):
            result[row["domain"]][row["split"]] = row["n"]
        for row in self.conn.execute("SELECT domain,COUNT(*) n FROM tasks JOIN probe USING(task_id) GROUP BY domain"):
            result[row["domain"]]["probe"] = row["n"]
        return result

    def omissions(self) -> list[dict]:
        return [dict(row) for row in self.conn.execute("SELECT source,reason,COUNT(*) n FROM omissions GROUP BY source,reason")]

    def coverage(self, cursor: dict[str, int]) -> dict:
        pools = self.counts()
        details = {}
        for domain in DOMAINS:
            total = pools[domain]["evolve_train"]
            used = cursor.get(domain, 0)
            if not 0 <= used <= total:
                raise ValueError("Coverage cursor outside declared pool")
            details[domain] = {"available_train_tasks": total, "unique_tasks_scheduled": used,
                               "scheduled_fraction": used / total if total else None}
        return {"domains": details, "all_tasks_scheduled": all(v["unique_tasks_scheduled"] == v["available_train_tasks"] for v in details.values())}


def _final_public_texts(row: dict, domain: str) -> list[str]:
    """Conservative public-prompt extraction; never scans answer/test/function fields."""
    keys = ("question_content", "prompt", "problem", "text") if domain == "code" else ("question", "query", "task", "prompt", "instruction")
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return [value]
        if domain == "tool_use" and isinstance(value, list):
            texts = []
            stack = list(value)
            while stack:
                message = stack.pop(0)
                if isinstance(message, list):
                    stack[0:0] = message
                elif isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), str):
                    texts.append(message["content"])
            if texts:
                return texts + (["\n".join(texts)] if len(texts) > 1 else [])
    for key in ("record", "data", "item"):
        if isinstance(row.get(key), dict):
            texts = _final_public_texts(row[key], domain)
            if texts:
                return texts
    return []


def audit_code_tool_leakage(source_manifest: str | Path, output_path: str | Path | None = None) -> dict:
    """Independent read-only audit of prepared sources, separate from Task/Meta.

    The new audit is conservative exact public-text/ID matching. Earlier prepared
    Code near-duplicate filtering is referenced by its evidence hash and is not
    relabeled as a newly executed semantic audit. No final labels are exported.
    """
    catalog_path = Path(source_manifest)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    train_hashes: dict[tuple[str, str], list[dict]] = {}
    env_ids = {"train": set(), "validation": set()}
    code_ids = {"train": set(), "validation": set()}
    evidence = []
    for item in catalog["datasets"]:
        if item["role"] not in {"train", "validation"} or item["domain"] not in {"code", "tool_use"}:
            continue
        path = Path(item["local_path"])
        evidence.append({"dataset_name": item["dataset_name"], "role": item["role"], "path": str(path), "sha256": sha256_file(path)})
        for _, _, row in jsonl_rows(path):
            domain = item["domain"]
            if domain == "tool_use":
                env_ids[item["role"]].add(str(row.get("canonical_env_id", row.get("env_id"))))
                if "task" not in row:  # environment metadata is indexed by environment ID only
                    continue
            prompt = _prompt(row, domain)
            qhash = content_hash(prompt)
            if domain == "code":
                code_ids[item["role"]].add(qhash)
            train_hashes.setdefault((domain, qhash), []).append({"source": _source_name(item), "role": item["role"], "task_hash": qhash})
    benchmark_results, matches = [], []
    for item in catalog["datasets"]:
        if item["role"] != "final_test" or item["domain"] not in {"code", "tool_use"}:
            continue
        path = Path(item["local_path"])
        count, parsed, matched = 0, 0, 0
        unknown_schemas = set()
        for _, _, row in jsonl_rows(path):
            count += 1
            texts = _final_public_texts(row, item["domain"])
            if not texts:
                unknown_schemas.add(tuple(sorted(row)))
                continue
            parsed += 1
            row_matches = []
            for text in texts:
                row_matches.extend(train_hashes.get((item["domain"], content_hash(text)), []))
            if row_matches:
                matched += 1
                matches.append({"benchmark": item["dataset_name"], "final_record_index": count - 1, "training_matches": row_matches})
        benchmark_results.append({"benchmark": item["dataset_name"], "domain": item["domain"], "source_sha256": sha256_file(path),
            "total_records": count, "parsed_public_prompt_records": parsed, "exact_overlapping_records": matched,
            "unparsed_schema_keys": [list(keys) for keys in sorted(unknown_schemas)],
            "status": "completed_exact_audit" if parsed == count else "partial_public_schema_coverage"})
    root = Path(catalog.get("data_root", catalog_path.parent.parent))
    upstream = []
    for relative in ("train/code/deepcoder_taco/contamination_report.json", "train/code/deepcoder_taco/dedup_report.json",
                     "train/code/deepcoder_taco/split_manifest.json", "train/tool_use/envscaler/split_manifest.json",
                     "manifests/code_verification.json", "manifests/tools_verification.json"):
        path = root / relative
        if path.is_file():
            upstream.append({"path": str(path), "sha256": sha256_file(path)})
    result = {"schema_version": 1, "decision_source": "independent_data_audit", "source_manifest_sha256": sha256_file(catalog_path),
              "method": "NFKC lowercase whitespace normalization; exact public prompt hashes and canonical environment groups",
              "near_duplicate_method": "not_reexecuted; prior prepared Code filtering evidence hashes recorded",
              "environment_train_dev_overlap": sorted(env_ids["train"] & env_ids["validation"]),
              "code_train_dev_overlap_count": len(code_ids["train"] & code_ids["validation"]),
              "benchmarks": benchmark_results, "exact_cross_final_matches": matches, "source_evidence": evidence,
              "prepared_audit_evidence": upstream, "final_answers_exported": False}
    result["status"] = "failed_overlap" if matches or result["environment_train_dev_overlap"] or result["code_train_dev_overlap_count"] else (
        "partial" if any(row["status"] != "completed_exact_audit" for row in benchmark_results) else "completed_exact_audit")
    if output_path:
        target = Path(output_path)
        if target.exists():
            raise FileExistsError("Audit evidence already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
