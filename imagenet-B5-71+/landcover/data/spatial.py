"""Spatial operations share one implementation for HWC images and CHW teachers."""
import random
import numpy as np
from mmseg.datasets.builder import PIPELINES


def _transform_fields(sample, operations, teacher=False):
    layouts = [("img", (0, 1))]
    if teacher:
        layouts.extend((key, (1, 2)) for key in
                       ("teacher_probs", "teacher_reliability") if key in sample)
    else:
        layouts.extend((key, (0, 1)) for key in sample.get("seg_fields", []))
    for key, axes in layouts:
        array = sample[key]
        for kind, value in operations:
            if kind == "rotate":
                array = np.rot90(array, value, axes=axes)
            else:
                array = np.flip(array, axis=axes[value])
        sample[key] = np.ascontiguousarray(array)
    return sample


@PIPELINES.register_module()
class QuarterTurn:
    def __init__(self, prob=0.5):
        self.prob = float(prob)

    def __call__(self, results):
        if random.random() >= self.prob:
            return results
        _transform_fields(results, [("rotate", random.randint(1, 3))])
        results["img_shape"] = results["img"].shape
        return results

    def __repr__(self):
        return f"{type(self).__name__}(prob={self.prob})"


@PIPELINES.register_module()
class VerticalReflection:
    def __init__(self, prob=0.5):
        self.prob = float(prob)

    def __call__(self, results):
        if random.random() < self.prob:
            _transform_fields(results, [("flip", 0)])
        return results

    def __repr__(self):
        return f"{type(self).__name__}(prob={self.prob})"


@PIPELINES.register_module()
class TeacherGeometry:
    def __init__(self, rotate90_prob=0.5, horizontal_prob=0.5, vertical_prob=0.5):
        self.rotate90_prob = float(rotate90_prob)
        self.horizontal_prob = float(horizontal_prob)
        self.vertical_prob = float(vertical_prob)

    def __call__(self, results):
        operations = []
        if random.random() < self.rotate90_prob:
            operations.append(("rotate", random.randint(1, 3)))
        if random.random() < self.horizontal_prob:
            operations.append(("flip", 1))
        if random.random() < self.vertical_prob:
            operations.append(("flip", 0))
        _transform_fields(results, operations, teacher=True)
        results.update(img_shape=results["img"].shape, flip=False, flip_direction=None)
        return results
