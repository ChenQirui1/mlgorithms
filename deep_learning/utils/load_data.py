from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Resize, Normalize


def load_mnist():

    training_data = MNIST(
        root="../data",
        train=True,
        download=True,
        transform=Compose(
            [Resize(32), ToTensor(), Normalize(mean=(0.1307,), std=(0.3081,))]
        )
    )

    dataloader = DataLoader(training_data, batch_size=64,
                            shuffle=True)

    return dataloader


if __name__ == "__main__":
    load_mnist()
