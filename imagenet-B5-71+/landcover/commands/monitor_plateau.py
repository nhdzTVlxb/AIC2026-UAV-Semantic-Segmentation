from __future__ import annotations
from landcover.cli import command_arguments
import argparse
import json
import os
import signal
import time
from pathlib import Path
from landcover.commands._monitor_plateau_support import process_alive, evaluation_records
from landcover.bootstrap import ROOT, prepare_runtime
from landcover.artifacts import file_digest, write_record
prepare_runtime()

def execute() -> int:
    arguments, options = command_arguments('monitor_plateau')
    fastest = options.baseline
    stale = 0
    seen = set()
    print(f'watching pid={options.pid} baseline={fastest:.6f} patience={options.patience} min_delta={options.min_delta:.6f}', flush=True)
    while True:
        for record in evaluation_records(options.work_dir):
            key = (int(record.get('epoch', 0)), int(record.get('iter', 0)))
            if key in seen:
                continue
            seen.add(key)
            score = float(record['mIoU'])
            epoch = int(record['epoch'])
            if score > fastest + options.min_delta:
                fastest = score
                stale = 0
                tensor_state = 'improved'
            else:
                stale += 1
                tensor_state = 'no-improvement'
            print(f'epoch={epoch} mIoU={score:.6f} best={fastest:.6f} stale={stale}/{options.patience} state={tensor_state}', flush=True)
            if stale >= options.patience:
                print('patience reached; terminating this training stage', flush=True)
                try:
                    os.killpg(options.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    try:
                        os.kill(options.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                return 10
        if not process_alive(options.pid):
            break
        time.sleep(options.poll_seconds)
    print('training process exited before patience was reached', flush=True)
    return 0
if __name__ == '__main__':
    execute()
