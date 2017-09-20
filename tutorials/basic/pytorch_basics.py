import torch
import torchvision
from torch import nn
import numpy as np
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable

# Basic autograd ex 1

x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

y = w * x + b
y.backward()

print(x.grad)
print(w.grad)
print(b.grad)

# Basic autograd ex 2

x = Variable(torch.randn(5, 3))
y = Variable(torch.randn(5, 2))

linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('b: ', linear.bias)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

pred = linear(x)
loss = criterion(pred, y)
print('loss: ', loss.data[0])

print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

loss.backward()

print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

optimizer.step()

pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.data[0])


# Loading data from numpy
a = np.array([[1, 2], [3, 4]])
b = torch.from_numpy(a)
c = b.numpy()

# Implementing the input pipeline
train_dataset = datasets.CIFAR10(
    root='../data/',
    train=True,
    transform=transforms.ToTensor(),
    download=False
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=100,
    shuffle=True,
    num_workers=2
)

data_iter = iter(train_loader)

images, labels = data_iter.next()

for images, labels in train_loader:
    pass
    # print(images.shape)

# Input pipeline for custom dataset
class CustomDataset(data.Dataset):
    def __init__(self):
        # TODO:
        # 1. Initialize file path or list of file names
        pass

    def __getitem__(self, index):
        # TODO:
        # 1. Read one data from file
        # 2. Preprocess the data
        # 3. Return a data pair (e.b. image and label)
        pass

    def __len__(self):
        return 0

custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(
    dataset=custom_dataset,
    batch_size=100,
    shuffle=True,
    num_workers=2
)

# Using pretrained resnet.
resnet = torchvision.models.resnet18(pretrained=True)

for param in resnet.parameters():
    param.require_grad = False

resnet.fc = nn.Linear(resnet.fc.in_features, 100)

images = Variable(torch.randn(10, 3, 256, 256))
outputs = resnet(images)
print(outputs.size())

# Save entire model
torch.save(resnet, 'model.pkl')
model = torch.load('model.pkl')

# Save and load only the model params
torch.save(resnet.state_dict(), 'params.pkl')
resnet.load_state_dict(torch.load('33params.pkl'))
