"""
Detials
"""
# imports
from detectron2.engine.hooks import HookBase
import detectron2.utils.comm as comm

import torch
import numpy as np 
from contextlib import contextmanager

## hookbase classes
#    def __init__(self, eval_period, evaluator, cfg):
#        self._period = eval_period
#        self._evaluator = evaluator
#        self.cfg = cfg
#        self.best_map = 0.0
#    
#    def after_step(self):
#        """ custom eval hook """
#        current_iter = self.trainer.iter
#        # if current iter is an eval period or last iteration
#        if current_iter % self._period == 0 or current_iter == self.max_iter:
#            data_loader = self.build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST) 
#            with inference_context(self.trainer.model), torch.no_grad():
#                results = inference_on_dataset(self.trainer.model, data_loader, self._evaluator)
#            print(results)

class EvalHook(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.

    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_period, eval_function, checkpointer, eval_after_train=True):
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
        self._period = eval_period
        self._func = eval_function
        self._eval_after_train = eval_after_train
        self._checkpointer = checkpointer
        self.best_map = 0.0

    def _do_eval(self):
        results = self._func()
        current_map = results["segm"]["AP"]
        if current_map > self.best_map:
            self.best_map = current_map
            self._checkpointer.save("best_model")
            print("best models saved at mAP: {}".format(self.best_map))

        #if results:
        #    assert isinstance(
        #        results, dict
        #    ), "Eval function must return a dict. Got {} instead.".format(results)
        #
        #    flattened_results = flatten_results_dict(results)
        #    for k, v in flattened_results.items():
        #        try:
        #            v = float(v)
        #        except Exception as e:
        #            raise ValueError(
        #                "[EvalHook] eval_function should return a nested dict of float. "
        #                "Got '{}: {}' instead.".format(k, v)
        #            ) from e
        #    self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
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
