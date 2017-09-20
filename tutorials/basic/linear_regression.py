import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Hyper params
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

x_train = np.array([
    [3.3],
    [4.4],
    [5.5],
    [6.71]
], dtype=np.float32)

y_train = np.array([
    [1.7],
    [2.76],
    [2.09],
    [3.19],
], dtype=np.float32)


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression(input_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    inputs = Variable(torch.from_numpy(x_train))
    targets = Variable(torch.from_numpy(y_train))

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print('Epoch [%d/%d], Loss: %.4f' %(epoch + 1, num_epochs, loss.data[0]))

predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
plt.plot(x_train, y_train, 'ro', label='Original Data')
plt.plot(x_train, predicted, label='Fitted Line')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'model.pkl')
