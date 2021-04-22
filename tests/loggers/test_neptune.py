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
from unittest.mock import MagicMock, patch

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger
from tests.helpers import BoringModel


@patch('pytorch_lightning.loggers.neptune.neptune')
def test_neptune_online(neptune):
    logger = NeptuneLogger(api_key='test', project='project')

    created_run = neptune.init()

    # It's important to check if the internal variable _experiment was initialized in __init__.
    # Calling logger.experiment would cause a side-effect of initializing _experiment,
    # if it wasn't already initialized.
    assert logger._run_instance is None
    _ = logger.experiment
    assert logger._run_instance == created_run
    assert logger.name == created_run['sys/name'].fetch()
    assert logger.version == created_run['sys/id'].fetch()


@patch('pytorch_lightning.loggers.neptune.neptune')
def test_neptune_additional_methods(neptune):
    logger = NeptuneLogger(api_key='test', project='project')

    run_mock = MagicMock()
    neptune.init.return_value = run_mock

    logger.experiment['key1'].log(torch.ones(1))
    run_mock.__getitem__.assert_called_once_with('key1')
    run_mock.__getitem__().log.assert_called_once_with(torch.ones(1))


@patch('pytorch_lightning.loggers.neptune.neptune')
def test_neptune_leave_open_experiment_after_fit(neptune, tmpdir):
    """Verify that neptune experiment was closed after training"""
    model = BoringModel()

    def _run_training(logger):
        logger._run_instance = MagicMock()
        logger._run_instance.__getitem__.return_value.fetch.return_value = 'exp-name'
        trainer = Trainer(
            default_root_dir=tmpdir,
            max_epochs=1,
            limit_train_batches=0.05,
            logger=logger,
        )
        assert trainer.log_dir is None
        trainer.fit(model)
        assert trainer.log_dir is None
        return logger

    logger_close_after_fit = _run_training(NeptuneLogger(api_key='test', project='project'))
    assert logger_close_after_fit.experiment.stop.call_count == 1

    logger_open_after_fit = _run_training(NeptuneLogger(api_key='test', project='project', close_after_fit=False))
    assert logger_open_after_fit.experiment.stop.call_count == 0


def _assert_legacy(callback, *args, **kwargs):
    try:
        callback(*args, **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError("Should throw `ValueError`")


def test_legacy_init_kwargs():
    legacy_neptune_init_kwargs = [
        'project_name',
        'offline_mode',
        'experiment_name',
        'experiment_id',
        'params',
        'properties',
        'upload_source_files',
        'abort_callback',
        'logger',
        'upload_stdout',
        'upload_stderr',
        'send_hardware_metrics',
        'run_monitoring_thread',
        'handle_uncaught_exceptions',
        'git_info',
        'hostname',
        'notebook_id',
        'notebook_path',
    ]
    for legacy_kwarg in legacy_neptune_init_kwargs:
        _assert_legacy(
            NeptuneLogger,
            **{legacy_kwarg: None}
        )


def test_legacy_functions():
    logger = NeptuneLogger(api_key='test', project='project')

    # test all  functions
    _assert_legacy(logger.log_metric)
    _assert_legacy(logger.log_text)
    _assert_legacy(logger.log_image)
    _assert_legacy(logger.log_artifact)
    _assert_legacy(logger.set_property)
    _assert_legacy(logger.append_tags)

    # test random args
    _assert_legacy(logger.log_metric, 42, foo="bar")
