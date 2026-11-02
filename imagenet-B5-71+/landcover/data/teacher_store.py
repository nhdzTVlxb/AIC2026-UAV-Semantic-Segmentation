"""Decode the frozen teacher's quantized distributions and optional reliability."""
import os
from pathlib import Path
import mmcv
import numpy as np
from mmcv.parallel import DataContainer
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines.formatting import to_tensor


def _decode_distribution(quantized, height, width):
    values = quantized.transpose(1, 2, 0)
    if values.shape[:2] != (height, width):
        values = mmcv.imresize(values, (width, height), interpolation="bilinear")
    values = values.astype(np.float32) / 255.0
    values /= np.maximum(values.sum(axis=2, keepdims=True), 1e-6)
    return np.ascontiguousarray(values.transpose(2, 0, 1))


@PIPELINES.register_module()
class ReadTeacherDistribution:
    def __init__(self, prob_dir=None):
        fallback = "/data/pseudo/stage4/probs_q8"
        self.prob_dir = Path(prob_dir or os.environ.get("AIC_PSEUDO_PROB_ROOT", fallback))

    def __call__(self, results):
        identity = Path(results["filename"]).stem
        archive = self.prob_dir / (identity + ".npz")
        if not archive.is_file():
            raise FileNotFoundError(archive)
        with np.load(archive) as record:
            quantized = record["probs_q8"]
            reliability = record["reliability_q8"] if "reliability_q8" in record else None
        if quantized.ndim != 3 or quantized.shape[0] != 8:
            raise ValueError(f"invalid teacher probability shape for {identity}: {quantized.shape}")
        height, width = results["img"].shape[:2]
        results["teacher_probs"] = _decode_distribution(quantized, height, width)
        if reliability is not None:
            if reliability.ndim == 3 and reliability.shape[0] == 1:
                reliability = reliability[0]
            if reliability.ndim != 2:
                raise ValueError(f"invalid teacher reliability shape for {identity}: {reliability.shape}")
            if reliability.shape != (height, width):
                reliability = mmcv.imresize(reliability, (width, height), interpolation="nearest")
            results["teacher_reliability"] = np.ascontiguousarray(
                reliability.astype(np.float32)[None] / 255.0)
        return results

    def __repr__(self):
        return f"{type(self).__name__}(prob_dir={self.prob_dir!s})"


@PIPELINES.register_module()
class PackTeacherDistribution:
    def __call__(self, results):
        for key in ("teacher_probs", "teacher_reliability"):
            if key in results:
                values = np.ascontiguousarray(results[key].astype(np.float32))
                results[key] = DataContainer(to_tensor(values), stack=True)
        return results

    def __repr__(self):
        return type(self).__name__
