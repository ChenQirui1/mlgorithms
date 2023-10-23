import numpy as np


def InputNormalise(X: np.ndarray, dist=None):
    '''
    dist: distribution of the individual columns to transform
    tuple(mean(array), std(array))

    '''
    # if the distributin is not provided,
    # calculate mean and std from sample

    if dist == None:
        n = X.shape[0]
        mean = (1/n) * np.sum(X, axis=0)
        std = np.sqrt(1/(n-1) * np.sum((X - mean)**2, axis=0))

    new_X = (X - mean) / std

    return new_X, mean, std


class BatchNorm:
    def __init__(self) -> None:
        self.Z_norm = None
        self.gamma
        self.beta
        self.mean
        self.var

    def update(self):
        # self.norm_gamma = self.norm_gamma - self.lr * \
        #     V_w_corr/np.sqrt(S_w_corr+self.eps) + reg
        # self.norm_beta = self.norm_beta - self.lr * \
        #     V_w_corr/np.sqrt(S_w_corr+self.eps) + reg

        pass

    def forward(self, Z):
        # batch norm

        m = self.Z.shape[1]

        mean = (1/m) * np.sum(self.Z, axis=0)
        var = 1/(m)*np.sum((self.Z - mean)**2, axis=0)
        Z_norm = (self.Z - mean) / np.sqrt(var + self.eps)
        self.Z = self.norm_gamma*Z_norm + self.norm_beta

    def pred():
        pass


if __name__ == '__main__':
    array = np.random.rand(10, 3)
    print(InputNormalise(array))
