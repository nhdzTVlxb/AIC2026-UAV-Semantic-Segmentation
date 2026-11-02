import argparse
import hashlib
import os
import urllib.request
from pathlib import Path
from scale_fusion.workspace_state import PUBLIC_FILE, PUBLIC_SHA, digest
URL = 'https://huggingface.co/skm-xydxtd/loveda-intern-xl-imp-mtp-iter_80000/resolve/2328c3f0857162b7990aa34f12257aaaaf16ba67/loveda-intern-xl-imp-mtp-iter_80000.pth'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=PUBLIC_FILE)
    args = parser.parse_args()
    path = args.cache
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path) != PUBLIC_SHA:
            raise ValueError('Existing public initialization SHA mismatch')
        print('Verified public initialization', path)
        return
    temporary = path.with_name(path.name + '.partial')
    request = urllib.request.Request(URL, headers={'User-Agent': 'aic3-public-initialization'})
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as output:
        if response.status != 200:
            raise ValueError('Unexpected public download status')
        for block in iter(lambda: response.read(8 * 1024 * 1024), b''):
            output.write(block)
    if digest(temporary) != PUBLIC_SHA:
        raise ValueError('Downloaded public file SHA mismatch; partial preserved for inspection')
    temporary.replace(path)
    print('Downloaded and SHA-verified public initialization', path)
if __name__ == '__main__':
    main()
