import numpy as np

def InputNormalise(X: np.ndarray,dist=None):
    '''
    dist: distribution of the individual columns to transform
    tuple(mean(array), std(array))
    
    '''
    #if the distributin is not provided,
    #calculate mean and std from sample

    if dist == None:
        n = X.shape[0]
        mean = (1/n) * np.sum(X, axis=0)
        std = np.sqrt(1/(n-1) * np.sum((X - mean)**2,axis=0))

    new_X = (X - mean) / std
    
    return new_X, mean, std
    
if __name__ == '__main__':
    array = np.random.rand(10,3)
    print(InputNormalise(array))