"""
Detials
"""

# imports
from detectron2.engine.hooks import HookBase
import detectron2.utils.comm as comm

class EvalHook(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.

    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, cfg, eval_period, eval_function, checkpointer, eval_after_train=True):
        """
        Args:
            eval_period (int): the period to run `eval_function`. Set to 0 to
                not evaluate periodically (but still evaluate after the last iteration
                if `eval_after_train` is True).
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.
            eval_after_train (bool): whether to evaluate after the last iteration

        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        self.cfg = cfg
        self._period = eval_period
        self._func = eval_function
        self._eval_after_train = eval_after_train
        self._checkpointer = checkpointer
        self.stage = "pre_training"
        self.best_map = 0.0

    def _update_stage(self):
        current_iter = self.trainer.iter
        pre_train_iters = self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS
        burn_in_iters = self.cfg.PSEUDO_LABELING.BURN_IN_ITERS

        if current_iter < pre_train_iters:
            self.stage = "pre_training"
        elif current_iter < pre_train_iters + burn_in_iters:
            self.stage = "burn_in"
        else:
            self.stage = "distillation"

    def _do_eval(self):
        self._update_stage()  # Ensure the stage is updated before evaluation
        results = self._func()
        current_map = results["segm"]["AP"]
        if current_map > self.best_map:
            self.best_map = current_map
            self._checkpointer.save(f"{self.stage}_best_model")
            print(f"Best model saved at mAP: {self.best_map} for stage: {self.stage}")

        comm.synchronize()

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            # do the last eval in after_train
            if next_iter != self.trainer.max_iter:
                self._do_eval()

    def after_train(self):
        # This condition is to prevent the eval from running after a failed training
        if self._eval_after_train and self.trainer.iter + 1 >= self.trainer.max_iter:
            self._do_eval()
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self._func