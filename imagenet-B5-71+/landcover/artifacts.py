"""Shared hashing and atomic JSON persistence for command modules."""
import hashlib
import json


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_record(path, record):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    pending.replace(path)
