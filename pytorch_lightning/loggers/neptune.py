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

if _module_available("neptune"):
    from neptune import __version__

    _NEPTUNE_AVAILABLE = __version__.startswith('0.9.') or __version__.startswith('1.')
else:
    _NEPTUNE_AVAILABLE = False

if _NEPTUNE_AVAILABLE:
    try:
        from neptune import new as neptune
        from neptune.new.run import Run
        from neptune.new.exceptions import NeptuneLegacyProjectException, NeptuneOfflineModeFetchException
    except ImportError:
        import neptune
        from neptune.run import Run
        from neptune.exceptions import NeptuneLegacyProjectException, NeptuneOfflineModeFetchException
else:
    # needed for test mocks, and function signatures
    neptune, Run = None, None


class NeptuneLogger(LightningLoggerBase):
    r"""
    Log using `Neptune <https://neptune.ai>`_.

    Install it with pip:

    .. code-block:: bash

        pip install neptune-client

    Pass NeptuneLogger instance to the Trainer to log metadata with Neptune:

    .. testcode::

        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import NeptuneLogger

        # Arguments passed to the "NeptuneLogger" are used to create new run in neptune.
        # We are using an "api_key" for the anonymous user "neptuner" but you can use your own.
        neptune_logger = NeptuneLogger(
            api_key='ANONYMOUS',
            project='common/new-pytorch-lightning-integration',
            name='lightning-run',  # Optional
        )
        trainer = Trainer(max_epochs=10, logger=neptune_logger)

    Use the logger anywhere in your :class:`~pytorch_lightning.core.lightning.LightningModule` as follows:

    .. code-block:: python

        from neptune.new.types import File

        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                # log metrics
                acc = ...
                self.logger.experiment['train/acc'].log(acc)

                # log images
                img = ...
                self.logger.experiment['train/misclassified_images'].log(File.as_image(img))

            def any_lightning_module_function_or_hook(self):
                # log model checkpoint
                ...
                self.logger.experiment['checkpoints/epoch37'].upload('epoch=37.ckpt')

                # generic recipe
                metadata = ...
                self.logger.experiment['your/metadata/structure'].log(metadata)

    Check `Neptune docs <https://docs.neptune.ai/user-guides/logging-and-managing-runs-results/logging-runs-data>`_
    for more info about how to log various types metadata (scores, files, images, interactive visuals, CSVs, etc.).

    **Log after training is finished**

    If you want to log objects after the training is finished use ``close_after_fit=False``:

    .. code-block:: python

        neptune_logger = NeptuneLogger(
            ...
            close_after_fit=False,
            ...
        )
        trainer = Trainer(logger=neptune_logger)
        trainer.fit(model)

        # Log metadata after trainer.fit() is done, for example diagnostics chart
        from neptune.new.types import File
        from scikitplot.metrics import plot_confusion_matrix
        import matplotlib.pyplot as plt
        ...
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(y_true, y_pred, ax=ax)
        neptune_logger.experiment['test/confusion_matrix'].upload(File.as_image(fig))

    **Pass additional parameters to Neptune run**

    You can also pass `kwargs` to specify the run in the greater detail, like ``tags`` and ``description``:

    .. testcode::

        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import NeptuneLogger

        neptune_logger = NeptuneLogger(
            project='common/new-pytorch-lightning-integration',
            name='lightning-run',
            description='mlp quick run with pytorch-lightning',
            tags=['mlp', 'quick-run'],
            )
        trainer = Trainer(max_epochs=3, logger=neptune_logger)

    Check `run documentation <https://docs.neptune.ai/essentials/api-reference/run>`_
    for more info about additional run parameters.

    **Details about Neptune run structure**

    Runs can be viewed as nested dictionary-like structures that you can define in your code.
    Thanks to this you can easily organize your metadata in a way that is most convenient for you.

    The hierarchical structure that you apply to your metadata will be reflected later in the UI.

    You can organize this way any type of metadata - images, parameters, metrics, model checkpoint, CSV files, etc.

    See Also:
        You can read about `what object you can log to Neptune <https://docs.neptune.ai/user-guides/
        logging-and-managing-runs-results/logging-runs-data#what-objects-can-you-log-to-neptune>`_.
        Also check `example run <https://app.neptune.ai/o/common/org/new-pytorch-lightning-integration/e/NEWPL-8/all>`_
        with multiple type of metadata logged.

    Args:
        api_key: Optional.
            Neptune API token, found on https://neptune.ai upon registration.
            Read: `how to find and set Neptune API token <https://docs.neptune.ai/administration/security-and-privacy/
            how-to-find-and-set-neptune-api-token>`_.
            It is recommended to keep it in the `NEPTUNE_API_TOKEN`
            environment variable and then you can drop ``api_key=None``.
        project: Optional.
            Qualified name of a project in a form of "my_workspace/my_project" for example "tom/mask-rcnn".
            If ``None``, the value of `NEPTUNE_PROJECT` environment variable will be taken.
            You need to create the project in https://neptune.ai first.
        close_after_fit: Optional default ``True``.
            If ``False`` the run will not be closed after training
            and additional metrics, images or artifacts can be logged.
        name: Optional. Editable name of the run.
            Run name appears in the "all metadata/sys" section in Neptune UI.
        run: Optional. Default is ``None``. The ID of the existing run.
            If specified (e.g. 'ABC-42'), connect to run with `sys/id` in project_name.
            Input argument "name" will be overridden based on fetched run data.
        prefix: A string to put at the beginning of metric keys.
        \**kwargs: Additional arguments like ``tags``, ``description``, ``capture_stdout``, ``capture_stderr`` etc.
            used when run is created.

    Raises:
        ImportError:
            If required Neptune package in version >=0.9 is not installed on the device.
        TypeError:
            If configured project has not been migrated to new structure yet.
    """

    LOGGER_JOIN_CHAR = '/'

    def __init__(
            self,
            api_key: Optional[str] = None,
            project: Optional[str] = None,
            close_after_fit: Optional[bool] = True,
            name: Optional[str] = None,
            run: Optional[str] = None,
            prefix: str = '',
            **neptune_run_kwargs):
        if neptune is None:
            raise ImportError(
                'You want to use `neptune` in version >=0.9 logger which is not installed yet,'
                ' install it with `pip install "neptune-client>=0.9"`.'
            )
        super().__init__()
        self._project = project
        self._api_key = api_key
        self._neptune_run_kwargs = neptune_run_kwargs
        self._close_after_fit = close_after_fit
        self._name = name
        self._run_to_load = run  # particular id of exp to load e.g. 'ABC-42'
        self._prefix = prefix

        self._run_instance = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Run instance can't be pickled
        state['_run_instance'] = None
        return state

    @property
    @rank_zero_experiment
    def experiment(self) -> Run:
        r"""
        Actual Neptune run object. Allows you to use neptune logging features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule`.

        Example::

            class LitModel(LightningModule):
                def training_step(self, batch, batch_idx):
                    # log metrics
                    acc = ...
                    self.logger.experiment['train/acc'].log(acc)

                    # log images
                    img = ...
                    self.logger.experiment['train/misclassified_images'].log(File.as_image(img))
        """
        return self.run

    @property
    def run(self) -> Run:
        if self._run_instance is None:
            try:
                self._run_instance = neptune.init(
                    project=self._project,
                    api_token=self._api_key,
                    run=self._run_to_load,
                    name=self._name,
                    **self._neptune_run_kwargs,
                )
            except NeptuneLegacyProjectException as e:
                raise TypeError(f"""
                    Project {self._project} has not been imported to new structure yet.
                    You can still integrate it with `NeptuneLegacyLogger`.
                    """) from e

        return self._run_instance

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        r"""
        Log hyper-parameters to the run.

        Params will be logged using the ``param__`` scheme, for example: ``param__batch_size``, ``param__lr``.

        **Note**

        You can also log parameters by directly using the logger instance:
        ``neptune_logger.experiment['model/hyper-parameters'] = params_dict``.

        In this way you can keep hierarchical structure of the parameters.

        Args:
            params: `dict`.
                Python dictionary structure with parameters.

        Example::

            from pytorch_lightning.loggers import NeptuneLogger

            PARAMS = {'batch_size': 64,
                      'lr': 0.07,
                      'decay_factor': 0.97}

            neptune_logger = NeptuneLogger(api_key='ANONYMOUS',
                                           close_after_fit=False,
                                           project='common/new-pytorch-lightning-integration')

            neptune_logger.log_hyperparams(PARAMS)
        """
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        for key, val in params.items():
            self.run[f'param__{key}'] = val

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Union[torch.Tensor, float]], step: Optional[int] = None) -> None:
        """
        Log metrics (numeric values) in Neptune runs.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values.
            step: Step number at which the metrics should be recorded, currently ignored.
        """
        assert rank_zero_only.rank == 0, 'run tried to log from global_rank != 0'

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
        try:
            self.run.sync()
        except NeptuneOfflineModeFetchException:
            return 'offline-name'
        return self.run['sys/name'].fetch()

    @property
    def version(self) -> str:
        try:
            self.run.sync()
        except NeptuneOfflineModeFetchException:
            return 'offline-id-1234'
        return self.run['sys/id'].fetch()
