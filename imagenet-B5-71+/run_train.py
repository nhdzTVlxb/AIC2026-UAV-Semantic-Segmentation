"""Forward the training CLI to the pinned official InternImage trainer."""
import os
import runpy
from landcover.bootstrap import SEGMENTATION, prepare_runtime


def main():
    os.environ.setdefault("AIC_TRUSTED_CHECKPOINTS", "1")
    prepare_runtime(trusted=os.environ.get("AIC_TRUSTED_CHECKPOINTS") == "1")
    import landcover.data.catalog
    import landcover.data.spatial
    import landcover.data.teacher_store
    import landcover.data.crop_sampling
    import landcover.training.policies
    import landcover.training.warm_start
    runpy.run_path(str(SEGMENTATION / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
