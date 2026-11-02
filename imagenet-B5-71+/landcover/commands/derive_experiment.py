from __future__ import annotations
from landcover.cli import command_arguments
import argparse
import json
from pathlib import Path
from mmcv import Config
from landcover.bootstrap import ROOT, prepare_runtime
from landcover.artifacts import file_digest, write_record
prepare_runtime()

def execute() -> None:
    arguments, options = command_arguments('derive_experiment')
    if options.crop_size <= 0 or options.max_epochs <= 0:
        raise ValueError('crop size and max epochs must be positive')
    recipe = Config.fromfile(str(options.base_config))
    recipe.work_dir = str(options.work_dir)
    recipe.load_from = str(options.load_from)
    recipe.resume_from = None
    recipe.optimizer.lr = options.lr
    recipe.crop_size = (options.crop_size, options.crop_size)
    labeled_pipeline = recipe.data.train[0].pipeline
    labeled_pipeline[3].img_scale = (options.crop_size * 4, options.crop_size)
    labeled_pipeline[4].crop_size = (options.crop_size, options.crop_size)
    labeled_pipeline[9].size = (options.crop_size, options.crop_size)
    teacher_pipeline = recipe.data.train[1].pipeline
    teacher_pipeline[1].prob_dir = str(options.pseudo_root)
    teacher_pipeline[2].crop_size = (options.crop_size, options.crop_size)
    teacher_pipeline[6].size = (options.crop_size, options.crop_size)
    if options.val_split is not None:
        recipe.data.val.split = str(options.val_split)
    if options.accept_thresholds and options.threshold_report:
        raise ValueError('provide either --accept-thresholds or --threshold-report')
    if options.threshold_report:
        record = json.loads(options.threshold_report.read_text(encoding='utf-8'))
        class_limits = [float(value) for value in record['thresholds']]
    elif options.accept_thresholds:
        class_limits = [float(value) for value in options.accept_thresholds.split(',')]
    else:
        class_limits = None
    if class_limits is not None:
        if len(class_limits) != 8:
            raise ValueError('accept-thresholds must contain exactly 8 comma-separated values')
        if any((not 0.0 <= value <= 1.0 for value in class_limits)):
            raise ValueError('accept thresholds must be within [0, 1]')
        recipe.model.pseudo_cfg.accept_threshold = class_limits
    recipe.model.pseudo_cfg.reject_floor = options.reject_floor
    recipe.checkpoint_config = dict(by_epoch=not options.one_iter_smoke, interval=2 if options.one_iter_smoke else 1, max_keep_ckpts=1, save_optimizer=False)
    recipe.evaluation = dict(by_epoch=False if options.one_iter_smoke else True, interval=999999 if options.one_iter_smoke else 1, metric='mIoU', pre_eval=True, save_best='mIoU', rule='greater')
    if options.one_iter_smoke:
        recipe.runner = dict(type='IterBasedRunner', max_iters=1)
        recipe.log_config.interval = 1
    else:
        recipe.runner.max_epochs = options.max_epochs
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.work_dir.mkdir(parents=True, exist_ok=True)
    recipe.dump(str(options.output))
    print(f'config={options.output} work_dir={options.work_dir} crop={options.crop_size} lr={options.lr} max_epochs={options.max_epochs} smoke={options.one_iter_smoke}', flush=True)
if __name__ == '__main__':
    execute()
