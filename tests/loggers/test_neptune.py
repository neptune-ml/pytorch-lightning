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


@patch('pytorch_lightning.loggers.neptune.neptune_alpha')
def test_neptune_online(neptune_alpha):
    logger = NeptuneLogger(api_key='test', project_name='project')

    created_run = neptune_alpha.init()

    # It's important to check if the internal variable _experiment was initialized in __init__.
    # Calling logger.experiment would cause a side-effect of initializing _experiment,
    # if it wasn't already initialized.
    assert logger._run_instance is None
    _ = logger.experiment
    assert logger._run_instance == created_run
    assert logger.name == created_run['sys/name'].fetch()
    assert logger.version == created_run['sys/id'].fetch()


@patch('pytorch_lightning.loggers.neptune.neptune_alpha')
def test_neptune_additional_methods(neptune_alpha):
    logger = NeptuneLogger(api_key='test', project_name='project')

    run_mock = MagicMock()
    neptune_alpha.init.return_value = run_mock

    logger.experiment['key1'].log(torch.ones(1))
    run_mock.__getitem__.assert_called_once_with('key1')
    run_mock.__getitem__().log.assert_called_once_with(torch.ones(1))
    run_mock.reset_mock()


@patch('pytorch_lightning.loggers.neptune.neptune_alpha')
def test_neptune_leave_open_experiment_after_fit(neptune_alpha, tmpdir):
    """Verify that neptune experiment was closed after training"""
    model = BoringModel()

    def _run_training(logger):
        logger._run_instance = MagicMock()
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

    logger_close_after_fit = _run_training(NeptuneLogger(offline_mode=True))
    assert logger_close_after_fit.experiment.stop.call_count == 1

    logger_open_after_fit = _run_training(NeptuneLogger(offline_mode=True, close_after_fit=False))
    assert logger_open_after_fit.experiment.stop.call_count == 0
