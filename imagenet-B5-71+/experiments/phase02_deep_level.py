"""MMCV entry; values live in the shared declarative recipe catalog."""
from landcover.recipes import experiment
globals().update(experiment('phase02_deep_level'))
