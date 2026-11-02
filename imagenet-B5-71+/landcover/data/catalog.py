"""MMSeg dataset registrations for eight-class PNG imagery."""
from mmseg.datasets import CustomDataset
from mmseg.datasets.builder import DATASETS

LABEL_NAMES = ("Background", "Building", "Road", "Water", "Barren",
               "Vegetation", "Agricultural", "Vehicle")
LABEL_COLORS = [[0, 0, 0], [220, 20, 60], [128, 64, 128], [0, 128, 255],
                [210, 180, 140], [34, 139, 34], [255, 215, 0], [255, 0, 255]]


@DATASETS.register_module()
class LandCoverDataset(CustomDataset):
    CLASSES = LABEL_NAMES
    PALETTE = LABEL_COLORS

    def __init__(self, **options):
        super().__init__(img_suffix=".png", seg_map_suffix=".png",
                         reduce_zero_label=True, **options)


@DATASETS.register_module()
class TeacherImageDataset(LandCoverDataset):
    def get_ann_info(self, idx):
        return {}
