from abc import ABC

from torch import optim


class ConfigureOptimizersPool(ABC):
    def configure_optimizers(self):
        """
        return whatever optimizers we want here.
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def configure_optimizers_empty(self):
        return None

    def configure_optimizers_lbfgs(self):
        """
        return whatever optimizers we want here.
        :return: list of optimizers
        """
        optimizer = optim.LBFGS(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def configure_optimizers_multiple_optimizers(self):
        """
        return whatever optimizers we want here.
        :return: list of optimizers
        """
        # try no scheduler for this model (testing purposes)
        optimizer1 = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        optimizer2 = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer1, optimizer2

    def configure_optimizers_single_scheduler(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
        return [optimizer], [lr_scheduler]

    def configure_optimizers_multiple_schedulers(self):
        optimizer1 = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        optimizer2 = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler1 = optim.lr_scheduler.StepLR(optimizer1, 1, gamma=0.1)
        lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, 1, gamma=0.1)

        return [optimizer1, optimizer2], [lr_scheduler1, lr_scheduler2]

    def configure_optimizers_mixed_scheduling(self):
        optimizer1 = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        optimizer2 = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler1 = optim.lr_scheduler.StepLR(optimizer1, 4, gamma=0.1)
        lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, 1, gamma=0.1)

        return [optimizer1, optimizer2], \
            [{'scheduler': lr_scheduler1, 'interval': 'step'}, lr_scheduler2]

    def configure_optimizers_reduce_lr_on_plateau(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [lr_scheduler]