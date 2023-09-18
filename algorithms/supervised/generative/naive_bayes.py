import numpy as np
from ...utils import util


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
        self.n_labels = None

    def calc_feat_proba(self, x, y, mode):
        """
        Calculates the probabilities of a feature x given the class labels.

        Arg:
            x: values of feature x. Shape (m,) {0,1}
            y: values of label y.  Shape (m,) 
        """
        probas = []
        labels = np.unique(y)

        if mode == 'bernouill':
            # proba of the feature x appearing(x=1) when y = i
            for i in labels:
                proba = np.sum((x == 1) & (y == i))/np.sum(y == i)
                probas.append(proba)

        elif mode == 'categorical':
            for i in np.unique(x):
                probas_per_feat_val = []
                for j in labels:
                    # proba of the feature x is some value when y is some value
                    proba = np.sum((x == i) & (y == j))/np.sum(y == j)
                    probas_per_feat_val.append(proba)

                probas.append(probas_per_feat_val)

        return probas

    def calc_prior_proba(self, y):
        """
        Calculates the probabilities of label y

        Arg:
            y: values of label y.  Shape (m,) {0,1}
        """

        probas = []

        for i in np.unique(y):
            prior_proba = np.sum(y == i)/len(y)

            probas.append(prior_proba)

        return probas

    def predict(self, X):
        """Make a prediction given new inputs x.

        Args:
            X: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """

        # applying the
        # theta (m,2)
        # prior (2,)
        scores = np.dot(X, self.theta)*self.prior

        prediction = np.argmax(scores, axis=1)

        return prediction


class BernouillNB(BaseNB):
    """Bernouill implementation of naive bayes.

    Example usage:
        > clf = BernouillNB()
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

        m, n = X.shape

        probs = []

        for i in range(n):

            x = X[:, i]

            probs.append(self.calc_feat_proba(x, y, 'bernouill'))

        self.theta = np.array(probs)
        self.prior = np.array(self.calc_prior_proba(y))
        self.n_labels = len(np.unique(y))

        return self

# #broken not fix yet
# class CategoricalNB(BaseNB):
#     """Categorical implementation of naive bayes.

#     Example usage:
#         > clf = CategoricalNB()
#         > clf.fit(x_train, y_train)
#         > clf.predict(x_eval)

#     """

#     def fit(self, X, y):

#         m,n = X.shape

#         probs = []

#         for i in range(n):

#             x = X[:,i]

#             probs.append(self.calc_feat_proba(x,y,'categorical'))

#         self.theta = np.transpose(np.array(probs),axes=[1,2,0])
#         self.prior = np.array(self.calc_prior_proba(y))
#         self.n_labels = len(np.unique(y))

#         return self


#     def lookup(self,X):

#         m,n = np.shape(X)


#         # for feat in range(n):
#         #     feat_vals = X[:,feat]
#         #     for feat_category in np.unique(feat_vals):
#         #         for label in range(self.n_labels):
#         #             np.where(feat_vals == feat_category,self.theta[feat_category,label,feat],feat_vals)

#         prediction = []

#         for feat in range(n):
#             feat_vals = X[:,feat]
#             for feat_category in np.unique(feat_vals):
#                 feat_vals = np.where(feat_vals == feat_category,self.theta[feat_category,:,feat],feat_vals)
#             prediction.append(feat_vals)

#         return prediction


class MultinomialNB(BaseNB):
    def __init__(self):
        super().__init__()
        self.term_frequency_per_label = None

    def sum_term_frequency(self, X, y):
        freq_per_label = []
        labels = np.unique(y)
        for label in labels:
            freq_per_label.append(
                np.sum(np.sum(X, axis=1), where=(y == label)))

        return freq_per_label

    def calc_feat_proba(self, x, y):
        labels = np.unique(y)
        proba = []
        for label in labels:
            proba.append(np.sum(x, where=(y == label)) /
                         self.term_frequency_per_label[label])
        return proba

    def fit(self, X, y):
        m, n = X.shape

        self.term_frequency_per_label = self.sum_term_frequency(X, y)

        probas = []
        for i in range(n):
            x = X[:, i]
            probas.append(self.calc_feat_proba(x, y))

        self.theta = np.array(probas)
        self.prior = np.array(self.calc_prior_proba(y))
        self.n_labels = len(np.unique(y))

        return self
