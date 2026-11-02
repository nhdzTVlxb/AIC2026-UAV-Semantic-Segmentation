from __future__ import annotations
import argparse
import json
import os
import signal
import time
from pathlib import Path
from landcover.bootstrap import ROOT, prepare_runtime
from landcover.artifacts import file_digest, write_record
prepare_runtime()

def process_alive(pid: int) -> bool:
    stat_path = Path(f'/proc/{pid}/stat')
    try:
        tensor_state = stat_path.read_text(encoding='utf-8').rsplit(')', 1)[1].split()[0]
    except (FileNotFoundError, IndexError):
        return False
    return tensor_state != 'Z'

def evaluation_records(work_dir: Path) -> list[dict]:
    entries = []
    for location in work_dir.glob('*.log.json'):
        try:
            with location.open('r', encoding='utf-8') as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if 'mIoU' in record and 'epoch' in record:
                        entries.append(record)
        except OSError:
            continue
    entries.sort(key=lambda item: (int(item.get('epoch', 0)), int(item.get('iter', 0))))
    unique = {}
    for record in entries:
        key = (int(record.get('epoch', 0)), int(record.get('iter', 0)))
        unique[key] = record
    return [unique[key] for key in sorted(unique)]
