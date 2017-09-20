from templates.bases import Base, Trainer
from templates.config import Config
import torchnet as tnt


class ConcreteTrainer(Base, Trainer):
    def __init__(self):
        model, criterion, optim = self.set_up()
        config = Config()
        Base.__init__(self, config)
        Trainer.__init__(self, model, criterion, optim)

        self.meters = [tnt.meter.AverageValueMeter(), tnt.meter.ClassErrorMeter(accuracy=True)]
        self.batch_size = 60
        self.batch_workers = 2

    def set_up(self):
        # TODO : set up model, criterion, optimizer here.
        return 0, 0, 0

    def get_iterator(self, is_train):
        pass

    def get_loss_and_output(self, sample):
        pass

    def _print_information(self, prefix):
        pass

    def on_end_epoch(self, state):
        pass

    def on_forward(self, state):
        pass
