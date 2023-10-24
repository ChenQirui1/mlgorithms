from torch import nn
from torch.optim import Adam
import torch
from deep_learning.utils.train import train_loop
from deep_learning.utils.eval import test_loop

from deep_learning.utils.load_data import load_mnist


# with reference to Andrew Ng's coursera deep learning CNN course

# leNet-5
class LeNet5(nn.Module):
    """Input Dim: (batch_size,1,32,32)

    Args:
        nn (_type_): _description_
    """

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
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x


# alexnet
class AlexNet(nn.Module):
    """Input Dim: (batch_size,3,227,227)

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, (11, 11), stride=4),
            nn.BatchNorm2d(96),
            nn.MaxPool2d((3, 3), stride=2)
        )
        # padding = 5
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, (5, 5), padding=(2, 2)),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((3, 3), stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(384),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(384),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((3, 3), stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9216, 4096)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(4096, 1000)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x

# VGG-16


class VGG16(nn.Module):
    """Input Dim: (batch_size,3,224,224)

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=(1, 1)),
            nn.Conv2d(64, 64, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2), stride=2)
        )
        # padding = 5
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
            nn.Conv2d(128, 128, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), padding=(1, 1)),
            nn.Conv2d(256, 256, (3, 3), padding=(1, 1)),
            nn.Conv2d(256, 256, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), padding=(1, 1)),
            nn.Conv2d(512, 512, (3, 3), padding=(1, 1)),
            nn.Conv2d(512, 512, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), padding=(1, 1)),
            nn.Conv2d(512, 512, (3, 3), padding=(1, 1)),
            nn.Conv2d(512, 512, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(25088, 4096)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(4096, 1000)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    # test = torch.rand(10, 1, 32, 32)
    # mnist_data = load_mnist()
    x = torch.rand(1, 3, 224, 224)
    model = VGG16()

    # Initialise optimiser
    optim = Adam(model.parameters(), lr=1e-2)

    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()

    # train_loop(mnist_data, model, loss_fn, optim)
    # test_loop(mnist_data, model, loss_fn)
    # torch.save(model.state_dict(), "./deep_learning/models/LeNet5.pt")
    model.forward(x)
