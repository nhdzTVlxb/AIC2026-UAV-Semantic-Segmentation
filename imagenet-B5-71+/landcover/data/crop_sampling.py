"""Joint detail sampling for the student image and teacher fields."""
import random
import mmcv
import numpy as np
from mmseg.datasets.builder import PIPELINES

@PIPELINES.register_module()
class TeacherScaleCrop:
    """Jointly resize/crop image, teacher probabilities and reliability.

    Full-image resizing preserves global context. Crop mode keeps native-detail
    statistics and can preferentially sample Barren/near-Barren regions.
    """

    def __init__(self, crop_size=(768, 768), ratio_range=(0.75, 1.5), full_image_prob=0.35, barren_focus_prob=0.5, barren_margin=0.1, candidates=8):
        self.crop_size = tuple((int(value) for value in crop_size))
        self.ratio_range = tuple((float(value) for value in ratio_range))
        self.full_image_prob = float(full_image_prob)
        self.barren_focus_prob = float(barren_focus_prob)
        self.barren_margin = float(barren_margin)
        self.candidates = int(candidates)
        if len(self.crop_size) != 2 or min(self.crop_size) <= 0:
            raise ValueError('crop_size must contain two positive values')
        if not 0 < self.ratio_range[0] <= self.ratio_range[1]:
            raise ValueError('invalid ratio_range')
        if self.candidates <= 0:
            raise ValueError('candidates must be positive')

    @staticmethod
    def _scale_probabilities(teacher, width, height):
        hwc = teacher.transpose(1, 2, 0)
        hwc = mmcv.imresize(hwc, (width, height), interpolation='bilinear')
        teacher = np.maximum(hwc.transpose(2, 0, 1), 0.0)
        teacher /= np.maximum(teacher.sum(axis=0, keepdims=True), 1e-06)
        return teacher

    def __call__(self, results):
        image = results['img']
        teacher_distribution = results['teacher_probs']
        agreement = results.get('teacher_reliability')
        native_rows, native_columns = image.shape[:2]
        crop_rows, crop_columns = self.crop_size
        if random.random() < self.full_image_prob:
            image = mmcv.imresize(image, (crop_columns, crop_rows), interpolation='bilinear')
            teacher_distribution = self._scale_probabilities(teacher_distribution, crop_columns, crop_rows)
            scale_x, scale_y = (crop_columns / native_columns, crop_rows / native_rows)
            if agreement is not None:
                agreement = mmcv.imresize(agreement[0], (crop_columns, crop_rows), interpolation='nearest')[None]
        else:
            fraction = random.uniform(*self.ratio_range)
            rows, columns = image.shape[:2]
            scaled_rows = max(crop_rows, int(round(rows * fraction)))
            scaled_columns = max(crop_columns, int(round(columns * fraction)))
            scale_x, scale_y = (scaled_columns / native_columns, scaled_rows / native_rows)
            image = mmcv.imresize(image, (scaled_columns, scaled_rows), interpolation='bilinear')
            teacher_distribution = self._scale_probabilities(teacher_distribution, scaled_columns, scaled_rows)
            if agreement is not None:
                agreement = mmcv.imresize(agreement[0], (scaled_columns, scaled_rows), interpolation='nearest')[None]
            max_y, max_x = (scaled_rows - crop_rows, scaled_columns - crop_columns)
            crop_origins = [(random.randint(0, max_y), random.randint(0, max_x)) for _ in range(self.candidates)]
            if random.random() < self.barren_focus_prob:
                competing_scores = np.max(teacher_distribution[[0, 1, 2, 3, 5, 6, 7]], axis=0)
                boundary_region = np.abs(teacher_distribution[4] - competing_scores) <= self.barren_margin
                barren = teacher_distribution[4] >= competing_scores
                preferred_region = boundary_region | barren
                y, x = max(crop_origins, key=lambda position: int(preferred_region[position[0]:position[0] + crop_rows, position[1]:position[1] + crop_columns].sum()))
            else:
                y, x = crop_origins[0]
            image = image[y:y + crop_rows, x:x + crop_columns]
            teacher_distribution = teacher_distribution[:, y:y + crop_rows, x:x + crop_columns]
            if agreement is not None:
                agreement = agreement[:, y:y + crop_rows, x:x + crop_columns]
        results['img'] = np.ascontiguousarray(image)
        results['teacher_probs'] = np.ascontiguousarray(teacher_distribution)
        if agreement is not None:
            results['teacher_reliability'] = np.ascontiguousarray(agreement)
        results['img_shape'] = results['img'].shape
        results['pad_shape'] = results['img'].shape
        results['scale_factor'] = np.asarray([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
        results['keep_ratio'] = False
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(crop_size={self.crop_size}, ratio_range={self.ratio_range}, full_image_prob={self.full_image_prob}, barren_focus_prob={self.barren_focus_prob})'
