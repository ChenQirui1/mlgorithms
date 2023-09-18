import numpy as np
import utils.util as util
from scipy.stats import multivariate_normal
from supervised.discriminative.linear_model import LinearModel


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """

        # *** START CODE HERE ***

        def covariance_matrix(x):
            x_true = x[np.nonzero(y)]
            x_true = x_true - mu1
            x_true = np.repeat(x_true, n, axis=1) * np.tile(x_true, n)

            x_false = x[np.where(y == 0)]
            x_false = x_false - mu0
            x_false = np.repeat(x_false, n, axis=1) * np.tile(x_false, n)

            x_combine = np.vstack((x_true, x_false))
            total = np.sum(x_combine, axis=0)/m
            matrix = np.reshape(total, (n, n))

            return matrix

        m, n = np.shape(x)

        phi = sum(y)/m
        mu1 = np.sum(x[np.nonzero(y)], axis=0)/sum(y)
        mu0 = np.sum(x[np.where(y == 0)], axis=0)/(m-sum(y))
        sigma = covariance_matrix(x)

        theta = {
            "phi": phi,
            "mu0": mu0,
            "mu1": mu1,
            "sigma": sigma,
        }

        self.theta = theta

        return self.theta

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        phi = self.theta['phi']
        mu0 = self.theta['mu0']
        mu1 = self.theta['mu1']
        sigma = self.theta['sigma']

        prob_x_given_y_true = multivariate_normal.pdf(
            x[1, :], mean=mu1, cov=sigma)
        prob_x_given_y_false = multivariate_normal.pdf(
            x[1, :], mean=mu0, cov=sigma)

        # for prediction in probability, use the following
        # posterior = (prob_x_given_y_true*phi)/((prob_x_given_y_true*phi)+(prob_x_given_y_false*phi))

        prediction = []

        for feat_vec in x:
            prob_x_given_y_false = multivariate_normal.pdf(
                feat_vec, mean=mu0, cov=sigma)
            prob_x_given_y_true = multivariate_normal.pdf(
                feat_vec, mean=mu1, cov=sigma)

            prediction.append(
                np.argmax([prob_x_given_y_false, prob_x_given_y_true]))

        return prediction
