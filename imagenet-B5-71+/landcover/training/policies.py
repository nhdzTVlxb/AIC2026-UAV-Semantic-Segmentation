"""Enforce backbone trainability and omit optimizer state from checkpoints."""
from mmcv.runner import HOOKS, Hook


def _model(runner):
    wrapped = runner.model
    return wrapped.module if hasattr(wrapped, "module") else wrapped


@HOOKS.register_module()
class BackbonePolicy(Hook):
    def __init__(self, trainable_levels=()):
        self.trainable_levels = tuple(map(int, trainable_levels))

    def _apply(self, runner):
        backbone = _model(runner).backbone
        for tensor in backbone.parameters():
            tensor.requires_grad = False
        backbone.eval()
        for index in self.trainable_levels:
            component = backbone.levels[index]
            for tensor in component.parameters():
                tensor.requires_grad = True
            component.train()

    def before_run(self, runner):
        self._apply(runner)
        model = _model(runner)
        enabled = sum(t.numel() for t in model.parameters() if t.requires_grad)
        count = sum(t.numel() for t in model.parameters())
        runner.logger.info("Backbone levels=%s; trainable=%d/%d (%.2f%%)",
                           self.trainable_levels, enabled, count, enabled * 100.0 / count)

    def before_train_epoch(self, runner):
        self._apply(runner)


@HOOKS.register_module()
class ParametersOnlySaver(Hook):
    def before_run(self, runner):
        saver = runner.save_checkpoint
        if getattr(saver, "_aic_weight_only", False):
            return

        def omit_optimizer(*positional, **options):
            options["save_optimizer"] = False
            return saver(*positional, **options)

        omit_optimizer._aic_weight_only = True
        runner.save_checkpoint = omit_optimizer
        runner.logger.info("Checkpoint optimizer state disabled")
