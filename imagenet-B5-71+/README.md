# AIC 2026 InternImage-H + Mask2Former

This package contains the complete training, validation, six-view prediction, submission sealing, environment setup, and reproduction scripts for the AIC 2026 land-cover solution. It contains source code and fixed split files only. Put the dataset, checkpoints, teacher caches, and output directories on the host and mount them when the container runs.

## Contents

- `run_train.py`: training entry point for all eight stages.
- `run_predict.py`: Stage8 FP32 six-view Test2 prediction and ZIP sealing.
- `experiments/phase01_decoder.py` through `phase08_large_crop.py`: Stage1 through Stage8 recipes.
- `landcover/`: dataset registration, sampling, losses, training hooks, teacher-cache tools, validation, and submission tools.
- `vendor/InternImage/`: the InternImage-H and Mask2Former implementation used by the recipes.
- `vendor/mmcv_a100_v1.7.2/`: the pinned MMCV source used to build CUDA operators.
- `compat/a100/`: PyTorch 2.5/CUDA 12.4 compatibility patches.
- `splits/`: fixed train, validation, and Stage7 calibration splits.
- `scripts/`: stage selection, prediction, validation, and no-training smoke checks.

## Environment

The Dockerfile builds Python 3.10 with CUDA 12.4, PyTorch 2.5.1 + cu124, torchvision 0.20.1, MMCV 1.7.2 with CUDA operators, MMSegmentation 0.27.0, MMDetection 2.28.1, and DCNv3. The image is intended for an NVIDIA A100/A800 class GPU (`sm_80`) with the NVIDIA Container Toolkit.

The image build sets `AIC_SKIP_GPU_CHECK=1` because a Docker build host commonly has no attached GPU. The normal A100/A800 check runs when `setup_a100.sh` is run outside the image.

## Build and start

```bash
docker build -t aic2026-stage8 .
docker run --gpus all --rm -it \
  -v /host/aic_data:/data:ro \
  -v /host/checkpoints:/weights:ro \
  -v /host/aic_outputs:/outputs \
  -e AIC_DATA_ROOT=/data/train/train \
  -e AIC_TEST2_ROOT=/data/test_2/images \
  -e AIC_WORK_ROOT=/outputs/work_dirs \
  -e AIC_PRETRAINED_CHECKPOINT=/weights/mask2former_internimage_h_896_80k_cocostuff2ade20k.pth \
  aic2026-stage8
```

For a host environment instead of Docker, install the same versions and run `bash setup_a100.sh`. Set `AIC_ENV_DIR` if the virtual environment should live somewhere else.

## Stage recipes

| Stage | Recipe | Main purpose |
|---|---|---|
| 1 | `experiments/phase01_decoder.py` | frozen backbone, 640 crop |
| 2 | `experiments/phase02_deep_level.py` | unfreeze level 3, 640 crop |
| 3 | `experiments/phase03_upper_levels.py` | unfreeze levels 2 and 3, 768 crop |
| 4 | `experiments/phase04_consolidation.py` | 768 crop consolidation |
| 5 | `experiments/phase05_teacher_ema999.py` | fixed Stage4 teacher, hard labels plus KL, EMA 0.999 |
| 6 | `experiments/phase06_detail_sampling.py` | single-teacher detail sampling |
| 7 | `experiments/phase07_teacher_refresh.py` | refreshed teacher, 768 crop |
| 8 | `experiments/phase08_large_crop.py` | refreshed Stage7 teacher, 832 crop |

Run one stage with the selector script. It only chooses the recipe; checkpoint and work-directory paths remain explicit so that a run cannot silently use a different experiment.

```bash
scripts/train_stage.sh 1
scripts/train_stage.sh 2 --cfg-options load_from=/outputs/stage1/best.pth work_dir=/outputs/stage2
scripts/train_stage.sh 3 --cfg-options load_from=/outputs/stage2/best.pth work_dir=/outputs/stage3
scripts/train_stage.sh 4 --cfg-options load_from=/outputs/stage3/best.pth work_dir=/outputs/stage4
```

Stage5 through Stage8 also need the teacher probability directory selected with `AIC_PSEUDO_PROB_ROOT` (Stage6/7 use `AIC_PSEUDO_PROB_ROOT_VARIANT_1`; Stage8 uses `AIC_STAGE7_PSEUDO_ROOT`) and the previous stage checkpoint passed through `--cfg-options load_from=...`. The default Stage7 and Stage8 checkpoint paths are controlled by `AIC_STAGE6_CHECKPOINT` and `AIC_STAGE7_CHECKPOINT`; explicit `--cfg-options load_from=...` values take precedence. The fixed teacher and validation utilities are available as modules:

```bash
python -m landcover.commands.write_teacher --help
python -m landcover.commands.inspect_teacher --help
python -m landcover.commands.score_holdout --help
python -m landcover.commands.fit_confidence_limits --help
```

Use the exact paths produced by the previous stage and keep each stage in a new work directory. The recipes are in `landcover/recipes.json`, including their crop sizes, learning rates, data pipelines, and validation settings.

## Stage8 prediction and submission

The six-view entry point expects a checkpoint and a directory containing the 1,300 Test2 images. It refuses to overwrite a non-empty output directory, validates every PNG, and writes the submission at the following exact name:

`stage8_refresh_stage7_832_best_p1_test2.zip`

```bash
AIC_CHECKPOINT=/weights/stage8_refresh_stage7_832_best.pth \
AIC_TEST2_ROOT=/data/test_2/images \
AIC_OUTPUT_ROOT=/outputs/stage8_prediction \
scripts/predict_stage8.sh
```

The run keeps the recorded three-scale, horizontal-flip, FP32, 768 crop, 512 stride, Hann fusion, and label-range checks. A checkpoint containing EMA backups is exported before inference; use `--ema-decay 0.999` when the checkpoint was produced by the Stage5 EMA recipe.

To validate and seal an existing prediction directory explicitly:

```bash
scripts/validate_submission.sh \
  /outputs/stage8_prediction/test2_predictions \
  /data/test_2/images \
  /outputs/stage8_refresh_stage7_832_best_p1_test2.zip \
  /weights/stage8_refresh_stage7_832_best.pth
```

## Smoke check

After building the image, run `bash scripts/smoke_test.sh`. It checks that all eight stage configurations load, compiles the Python source, checks the CLI entry points, imports the CUDA-backed packages, and verifies the submission filename. It does not load data, load a checkpoint, train, validate a dataset, or generate predictions. In a GPU-enabled container, run `python scripts/smoke_cuda_ops.py` to check a tiny DCNv3 forward/backward operation without starting model training.
