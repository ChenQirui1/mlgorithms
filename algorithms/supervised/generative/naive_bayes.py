import numpy as np
from ...utils import util
from scipy.stats import multivariate_normal 


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    # x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    # x_eval,y_eval = util.load_dataset(eval_path, add_intercept=False)


class BaseNB():
    """Base implementation of naive bayes.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self):
        self.theta = []
        self.prior = []


    def predict(self, X):
        """Make a prediction given new inputs x.

        Args:
            X: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """

        # applying the 
        scores = np.dot(X,self.theta*self.prior)

        prediction = np.argmax(scores,axis=1)

        return prediction
    

    
class BernouillNB(BaseNB):
    """Base implementation of naive bayes.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, X, y):
        """Fit a naive bayes to training set given by x and y

        Args:
            X: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            <tuple>: Naive Bayes model parameters.
        """

        def cal_prob_feat(x,y):
            """
            Calculates the probabilities of a feature x given the class labels.

            Arg:
                x: values of feature x. Shape (m,) {0,1}
                y: values of label y.  Shape (m,) {0,1}
            """

            prob_1 = np.sum((x == 1) & (y == 1))/np.sum(y==1)

            prob_0 = np.sum((x == 1) & (y == 0))/np.sum(y==0)

            return (prob_0,prob_1)
        

        def cal_prior(y):
            """
            Calculates the probabilities of label y

            Arg:
                y: values of label y.  Shape (m,) {0,1}
            """

            prior_1 = np.sum(y)/len(y)

            prior_0 = np.sum(y==0)/len(y)

            return (prior_0,prior_1)
        

        m,n = X.shape

        probs = []

        for i in range(n):
            
            x = X[:,i]

            probs.append(cal_prob_feat(x,y))
    

        self.theta = np.array(probs)
        self.prior = np.array(cal_prior(y))

        return self