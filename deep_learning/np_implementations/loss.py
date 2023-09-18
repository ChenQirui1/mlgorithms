import numpy as np
from sklearn.metrics import mean_squared_error

def Cost(y_true,y_pred,norm,lambd,w):

    if norm == "l2":
        decay = (lambd/len(y_true)) * (w * w)
    else:
        decay = None

    return mean_squared_error(y_true,y_pred) + decay