# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Neptune Logger
--------------
"""
import logging
from argparse import Namespace
from typing import Any, Dict, Optional, Union

import torch

from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import _module_available, rank_zero_only

log = logging.getLogger(__name__)
_NEPTUNE_AVAILABLE = _module_available("neptune.new")

if _NEPTUNE_AVAILABLE:
    from neptune import new as neptune_alpha
    from neptune.new.run import Run
    from neptune.new.internal.init_impl import ASYNC, OFFLINE
else:
    # needed for test mocks, these tests shall be updated
    neptune_alpha, Run, ASYNC, OFFLINE = None, None, None, None


class NeptuneLogger(LightningLoggerBase):
    r"""
    Log using `Neptune <https://neptune.ai>`_.

    Install it with pip:

    .. code-block:: bash

        pip install neptune-client

    The Neptune logger can be used in the online mode or offline (silent) mode.
    To log experiment data in online mode, :class:`NeptuneLogger` requires an API key.
    In offline mode, the logger does not connect to Neptune.

    **ONLINE MODE**

    .. testcode::

        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import NeptuneLogger

        # arguments made to NeptuneLogger are passed on to the neptune.experiments.Experiment class
        # We are using an api_key for the anonymous user "neptuner" but you can use your own.
        neptune_logger = NeptuneLogger(
            api_key='ANONYMOUS',
            project_name='shared/pytorch-lightning-integration',
            experiment_name='default',  # Optional,
            params={'max_epochs': 10},  # Optional,
            tags=['pytorch-lightning', 'mlp']  # Optional,
        )
        trainer = Trainer(max_epochs=10, logger=neptune_logger)

    **OFFLINE MODE**

    .. testcode::

        from pytorch_lightning.loggers import NeptuneLogger

        # arguments made to NeptuneLogger are passed on to the neptune.experiments.Experiment class
        neptune_logger = NeptuneLogger(
            offline_mode=True,
            project_name='USER_NAME/PROJECT_NAME',
            experiment_name='default',  # Optional,
            params={'max_epochs': 10},  # Optional,
            tags=['pytorch-lightning', 'mlp']  # Optional,
        )
        trainer = Trainer(max_epochs=10, logger=neptune_logger)

    Use the logger anywhere in you :class:`~pytorch_lightning.core.lightning.LightningModule` as follows:

    .. code-block:: python

        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                # log metrics
                self.logger.experiment.log_metric('acc_train', ...)
                # log images
                self.logger.experiment.log_image('worse_predictions', ...)
                # log model checkpoint
                self.logger.experiment.log_artifact('model_checkpoint.pt', ...)
                self.logger.experiment.whatever_neptune_supports(...)

            def any_lightning_module_function_or_hook(self):
                self.logger.experiment.log_metric('acc_train', ...)
                self.logger.experiment.log_image('worse_predictions', ...)
                self.logger.experiment.log_artifact('model_checkpoint.pt', ...)
                self.logger.experiment.whatever_neptune_supports(...)

    If you want to log objects after the training is finished use ``close_after_fit=False``:

    .. code-block:: python

        neptune_logger = NeptuneLogger(
            ...
            close_after_fit=False,
            ...
        )
        trainer = Trainer(logger=neptune_logger)
        trainer.fit()

        # Log test metrics
        trainer.test(model)

        # Log additional metrics
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(y_true, y_pred)
        neptune_logger.experiment.log_metric('test_accuracy', accuracy)

        # Log charts
        from scikitplot.metrics import plot_confusion_matrix
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(y_true, y_pred, ax=ax)
        neptune_logger.experiment.log_image('confusion_matrix', fig)

        # Save checkpoints folder
        neptune_logger.experiment.log_artifact('my/checkpoints')

        # When you are done, stop the experiment
        neptune_logger.experiment.stop()

    See Also:
        - An `Example experiment <https://ui.neptune.ai/o/shared/org/
          pytorch-lightning-integration/e/PYTOR-66/charts>`_ showing the UI of Neptune.
        - `Tutorial <https://docs.neptune.ai/integrations/pytorch_lightning.html>`_ on how to use
          Pytorch Lightning with Neptune.

    Args:
        api_key: Required in online mode.
            Neptune API token, found on https://neptune.ai.
            Read how to get your
            `API key <https://docs.neptune.ai/python-api/tutorials/get-started.html#copy-api-token>`_.
            It is recommended to keep it in the `NEPTUNE_API_TOKEN`
            environment variable and then you can leave ``api_key=None``.
        project_name: Required in online mode. Qualified name of a project in a form of
            "namespace/project_name" for example "tom/minst-classification".
            If ``None``, the value of `NEPTUNE_PROJECT` environment variable will be taken.
            You need to create the project in https://neptune.ai first.
        offline_mode: Optional default ``False``. If ``True`` no logs will be sent
            to Neptune. Usually used for debug purposes.
        close_after_fit: Optional default ``True``. If ``False`` the experiment
            will not be closed after training and additional metrics,
            images or artifacts can be logged. Also, remember to close the experiment explicitly
            by running ``neptune_logger.experiment.stop()``.
        experiment_name: Optional. Editable name of the experiment.
            Name is displayed in the experiment’s Details (Metadata section) and
            in experiments view as a column.
        experiment_id: Optional. Default is ``None``. The ID of the existing experiment.
            If specified, connect to experiment with experiment_id in project_name.
            Input arguments "experiment_name", "params", "properties" and "tags" will be overriden based
            on fetched experiment data.
        prefix: A string to put at the beginning of metric keys.
        \**kwargs: Additional arguments like `params`, `tags`, `properties`, etc. used by
            :func:`neptune.Session.create_experiment` can be passed as keyword arguments in this logger.

    Raises:
        ImportError:
            If required Neptune package is not installed on the device.
    """

    LOGGER_JOIN_CHAR = '-'

    def __init__(
            self,
            project: Optional[str] = None,
            api_key: Optional[str] = None,
            run: Optional[str] = None,
            mode: str = ASYNC,
            close_after_fit: Optional[bool] = True,
            offline_mode: bool = False,
            **neptune_run_kwargs):
        if neptune_alpha is None:
            raise ImportError(
                'You want to use `neptune` in version >=0.9 logger which is not installed yet,'
                ' install it with `pip install "neptune-client>=0.9"`.'
            )
        super().__init__()
        self._project = project
        self._api_key = api_key
        self._run_to_load = run  # particular id of exp to load e.g. 'ABC-42'
        self._mode = mode
        self._neptune_run_kwargs = neptune_run_kwargs

        self._close_after_fit = close_after_fit
        self._run_mode = OFFLINE if offline_mode else ASYNC

        self._run_instance = None

        log.info(f'NeptuneLogger will work in {"offline" if self.offline_mode else "online"} mode')

    def __getstate__(self):
        # TODO: what is this function about
        state = self.__dict__.copy()
        return state

    @property
    def offline_mode(self):
        return self._run_mode == OFFLINE

    @property
    @rank_zero_experiment
    def experiment(self) -> Run:
        r"""
        Actual Neptune object. To use neptune features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_neptune_function()

        """

        # Note that even though we initialize self._experiment in __init__,
        # it may still end up being None after being pickled and un-pickled
        if self._run_instance is None:
            self._run_instance = neptune_alpha.init(
                project=self._project,
                api_token=self._api_key,
                run=self._run_to_load,
                mode=self._mode,
                **self._neptune_run_kwargs,
            )

        return self._run_instance

    @property
    def run(self) -> Run:
        return self.experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        for key, val in params.items():
            self.run[f'param__{key}'] = val

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Union[torch.Tensor, float]], step: Optional[int] = None) -> None:
        """
        Log metrics (numeric values) in Neptune experiments.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded, currently ignored
        """
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        metrics = self._add_prefix(metrics)
        for key, val in metrics.items():
            # `step` is ignored because Neptune expects strictly increasing step values which
            # Lighting does not always guarantee.
            self.experiment[key].log(val)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        super().finalize(status)
        if self._close_after_fit:
            self.run.stop()

    @property
    def save_dir(self) -> Optional[str]:
        # Neptune does not save any local files
        return None

    @property
    def name(self) -> str:
        if self.offline_mode:
            return 'offline-name'
        else:
            return self.run.name  # TODO

    @property
    def version(self) -> str:
        if self.offline_mode:
            return 'offline-id-1234'
        else:
            return str(self.run._uuid)
