"""Expand shared recipes while preserving lists, tuples and environment overrides."""
import json
import os
import re
from pathlib import Path
from landcover.bootstrap import ROOT


def experiment(identity):
    catalog = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))
    defaults = catalog["defaults"]

    def variable(match):
        binding = defaults[match.group(1)]
        fallback = binding["fallback"]
        if fallback == "@checkout":
            fallback = str(ROOT)
        return os.environ.get(binding["environment"], fallback)

    def expand(value):
        if isinstance(value, dict):
            if set(value) == {"$ref"}:
                return expand(catalog["shared"][value["$ref"]])
            if set(value) == {"$tuple"}:
                return tuple(expand(item) for item in value["$tuple"])
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            # Defaults can themselves refer to an environment variable, e.g. a checkpoint.
            while re.search(r"\$\{([A-Z0-9_]+)\}", value):
                value = re.sub(r"\$\{([A-Z0-9_]+)\}", variable, value)
            return value
        return value

    return expand(catalog["experiments"][identity])
