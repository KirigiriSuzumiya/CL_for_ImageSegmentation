import avalanche
import copy
from avalanche.training.utils import freeze_everything


from typing import Callable, Optional, Sequence, List, Union
import torch
from torch.nn import Module
from torch.optim import Optimizer

from avalanche.training.plugins.evaluation import (
    default_evaluator,
)
from avalanche.training.plugins import (
    SupervisedPlugin,
    EvaluationPlugin,
    LFLPlugin,
)
from avalanche.training.templates.strategy_mixin_protocol import CriterionType



class SGLFLPlugin(LFLPlugin):
    def __init__(self, lambda_e):
        """
        :param lambda_e: Euclidean loss hyper parameter
        """
        super().__init__(lambda_e)

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience
        and freeze the prev model and freeze the last layer of current model
        """

        self.prev_model = copy.deepcopy(strategy.model)

        freeze_everything(self.prev_model)


class SGLFL(avalanche.training.templates.SupervisedTemplate):
    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType,
        lambda_e: Union[float, Sequence[float]],
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        **base_kwargs
    ):
        lfl = SGLFLPlugin(lambda_e)
        if plugins is None:
            plugins = [lfl]
        else:
            plugins.append(lfl)

        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )