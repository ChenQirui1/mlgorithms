import numpy as np
import math
from sklearn.metrics import mean_squared_error
from np_implementations import nn


class Optimiser():

    def __init__(self, model, batch_size, X, y):
        self.model = model
        self.batch_size = batch_size

        self.X = np.array_split(X, self.batch_size, axis=1)
        self.y = np.array_split(y, self.batch_size)

        self.epochs = len(y) // self.batch_size

        self.tracker = 0

    def run(self):
        return self.minibgd(self.X, self.y)

    def minibgd(self, X, y):

        epochs = len(y) // self.batch_size

        for epoch in range(epochs):
            # A: (node_size,batch_size)
            A = self.model.forward(X[epoch])

            self.tracker += 1

            grad_A = -(y[epoch]/A) + (1-y[epoch])/(1-A)

            self.model.backward(grad_A)
