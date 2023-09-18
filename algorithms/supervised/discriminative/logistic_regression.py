import numpy as np
import utils.util as util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    clf.score(x_eval, y_eval)
    # prediction = clf.predict(x_eval)
    # with open(pred_path,"w+") as f:
    #     for line in prediction:
    #         f.write(str(line)+"\n")

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # declaring X and y
        X = np.array(x, dtype=np.float64)
        Y = np.array(y, dtype=np.float64)

        # declare the observation size and no's of features
        m, n = np.shape(X)
        self.theta = np.zeros(n)

        # vectorised function of g(x) based on design matrix X
        def g(X):
            array = np.empty(m, dtype=np.float64)

            for i in range(m):
                array[i] = 1/(1+np.exp(-(np.dot(self.theta, X[i]))))

            return array

        # loop for convergence
        prev_theta = np.ones(n)
        while np.linalg.norm(prev_theta - self.theta) > self.eps:
            #! -= operator does not work with vectorised form
            # hessian of likelihood func
            hessian = (1/m)*np.dot((X.T*(g(X)*(1-g(X)))), X)
            self.theta = self.theta - \
                np.dot(np.linalg.inv(hessian), (1/m)*np.dot(X.T, (g(X)-Y)))
            prev_theta = self.theta

        return self.theta

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        X = np.array(x)
        m, n = np.shape(X)
        output = np.empty(m, dtype=np.float64)
        raw_value = 0
        for i in range(m):
            raw_value = 1/(1+np.exp(-(np.dot(self.theta, X[i]))))
            output[i] = 1 if (raw_value > 0.5) else 0
        return output

    def score(self, x, y):
        prediction = self.predict(x)
        confusion_matrix = np.empty((2, 2))
        for i in range(len(prediction)):
            if prediction[i] == y[i] and y[i] == 1:
                confusion_matrix[0][0] += 1
            elif prediction[i] != y[i] and y[i] == 1:
                confusion_matrix[0][1] += 1
            elif prediction[i] != y[i] and y[i] == 0:
                confusion_matrix[1][0] += 1
            elif prediction[i] == y[i] and y[i] == 0:
                confusion_matrix[1][1] += 1

        # scoring by accuracy
        score = (confusion_matrix[0][0] +
                 confusion_matrix[1][1]) / np.sum(confusion_matrix)
        print(confusion_matrix)
        print(score)
        return score
