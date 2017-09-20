from bases import Base, Trainer
from config import Config
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
import torchnet as tnt
from torch.autograd import Variable

from models import Model


class ConcreteTrainer(Base, Trainer):
    learning_rate = 0.0001
    
    def __init__(self):
        model, criterion, optim = self.set_up()
        config = Config()
        Base.__init__(self, config)
        Trainer.__init__(self, model, criterion, optim)

        self.meters = [tnt.meter.AverageValueMeter()]
        self.batch_size = 1
        self.batch_workers = 2

    def set_up(self):
        # TODO : set up model, criterion, optimizer here.
        model = Model()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model, criterion, optimizer

    def get_iterator(self, is_train):
        dataset = datasets.MNIST(
            './data', train=is_train, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

        data = getattr(dataset, 'train_data' if is_train else 'test_data')
        labels = getattr(dataset, 'train_labels' if is_train else 'test_labels')
        tensor_datasets = tnt.dataset.TensorDataset([data, labels])
        return tensor_datasets.parallel(batch_size=self.batch_size, num_workers=self.batch_workers, shuffle=is_train)

    def get_loss_and_output(self, sample):
        inputs = Variable(sample[0].float())
        inputs = inputs.view(-1, 784)
        targets = Variable(torch.LongTensor(sample[1]))
        outputs, mu, logvar = self.model(inputs)
        bce_loss = self.criterion(outputs, inputs)

        # print(mu)
        kl_divergence_elem = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        kld_loss = torch.sum(kl_divergence_elem).mul_(-0.5)
        # print(kld_loss)
        return bce_loss + kld_loss, outputs

    def _print_information(self, prefix):
        print_str = '{0} loss: %.4f' % (self.meters[0].value()[0])
        print(print_str.format(prefix))

    def on_end_epoch(self, state):
        self._print_information('[Training]')
        self.reset_meters()
        self.engine.test(self.get_loss_and_output, self.get_iterator(False))
        self._print_information('[Validating]')
        self.save_model()

    def on_start(self, state):
        self.load_model()

    def on_forward(self, state):
        # if state['t'] % 10 == 0:
        #     print("{0}: {1}".format(state['t'], state['loss'].data[0]))
        self.meters[0].add(state['loss'].data[0])
