"""Construct command parsers from the shared, typed CLI contract."""
import argparse
import json
from pathlib import Path
from landcover.bootstrap import ROOT


def command_parser(identity):
    catalog = json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser()
    types = {'Path': Path, 'int': int, 'float': float, 'str': str}
    for entry in catalog[identity]:
        options = dict(entry['options'])
        if 'type' in options:
            options['type'] = types[options['type']]
        default = options.get('default')
        if isinstance(default, dict) and '$path' in default:
            spec = default['$path']
            location = ROOT.parent if spec['base'] == 'parent' else ROOT
            options['default'] = location.joinpath(*spec['parts'])
        parser.add_argument(*entry['names'], **options)
    return parser


def command_arguments(identity):
    parser = command_parser(identity)
    return parser, parser.parse_args()
