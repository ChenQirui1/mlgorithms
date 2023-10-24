from torch import nn
from torch.optim import Adam
import torch
from deep_learning.utils.train import train_loop
from deep_learning.utils.eval import test_loop

from deep_learning.utils.load_data import load_mnist


# leNet-5

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, (5, 5)),
            nn.BatchNorm2d(6),
            nn.MaxPool2d((2, 2))
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, (5, 5)),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((2, 2))
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(400, 120)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        # self.softmax = nn.Softmax(1)

    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        # x = self.softmax(x)

        return x


if __name__ == "__main__":
    # test = torch.rand(10, 1, 32, 32)
    mnist_data = load_mnist()
    model = LeNet5()

    # Initialise optimiser
    optim = Adam(model.parameters(), lr=1e-2)

    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()

    train_loop(mnist_data, model, loss_fn, optim)
    test_loop(mnist_data, model, loss_fn)
    torch.save(model.state_dict(), "./deep_learning/models/LeNet5.pt")
