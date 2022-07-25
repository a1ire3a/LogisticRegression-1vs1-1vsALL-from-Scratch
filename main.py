import numpy as np
from sklearn.preprocessing import StandardScaler
from statistics import mode
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[0:,1:3]
Y = iris.target

class LogisticRegression(object):

    def __init__(self, alpha=0.01, n_iteration=100,landa=0.001):
        self.alpha = alpha  # value in the object
        self.n_iter = n_iteration
        self.landa = landa

    def _sigmoid_function(self,x):
        value = 1 / (1 + np.exp(-x))
        return value

    def _cost_function(self, h, theta, y):
        m = len(y)
        cost = (1 / m) * (
                    np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + self.landa * np.dot(theta, theta))
        return cost

    def _gradient_descent(self, X, h, theta, y, m):
        gradient_value = (np.dot(X.T, (h - y)) + self.landa * theta) / m
        theta -= self.alpha * gradient_value
        return theta

    def fit_oneVSall(self, X,y):
        self.theta = []
        self.cost = []
        m = len(y)
        for i in np.unique(y):

            y_onevsall = np.where(y == i, 1, 0)
            theta = np.zeros(X.shape[1])
            cost = []
            for _ in range(self.n_iter):
                z = X.dot(theta)
                h = self._sigmoid_function(z)
                theta = self._gradient_descent(X, h, theta, y_onevsall, m)
                cost.append(self._cost_function(h, theta, y_onevsall))
            self.theta.append((theta, i))
            self.cost.append((cost, i))
        return self

    def fit_oneVSone(self, X, y):
        self.theta = []
        self.cost = []
        m = len(y)
        for i in np.unique(y):
            for j in np.unique(y):
                y_onevsone = []
                X_onevsone = []
                if j <= i:
                    continue
                else:
                    for k in range(m):
                        if y[k] == i:
                            y_onevsone.append(1)
                            X_onevsone.append(X[k])
                        elif y[k] == j:
                            y_onevsone.append(0)
                            X_onevsone.append(X[k])
                    y_onevsone = np.asarray(y_onevsone)
                    X_onevsone = np.asarray(X_onevsone)
                    theta = np.zeros(X_onevsone.shape[1])
                    cost = []
                    for _ in range(self.n_iter):
                        z = X_onevsone.dot(theta)
                        h = self._sigmoid_function(z)
                        theta = self._gradient_descent(X_onevsone, h, theta, y_onevsone, m)
                        cost.append(self._cost_function(h, theta, y_onevsone))
                    self.theta.append((theta, i, j))
                    self.cost.append((cost, i, j))
        return self

    def predict_oneVSall(self, X):
        X_predicted = [max((self._sigmoid_function(i.dot(theta)), c) for theta, c in self.theta)[1] for i in X]
        return X_predicted

    def predict_oneVSone(self, X):
        X_predicted = []
        for i in X:
            pred = []
            for th, c1, c2 in self.theta:
                if self._sigmoid_function(i.dot(th)) > 0.5:
                    pred.append(c1)
                else:
                    pred.append(c2)
            X_predicted.append(mode(pred))
        return X_predicted

        X_predicted = [max((self._sigmoid_function(i.dot(theta)), c1, c2) for theta, c1, c2 in self.theta)[1] for i in X]
        return X_predicted

    def score_oneVSall(self, X,y):
        score = sum(self.predict_oneVSall(X) == y) / len(y)
        return score

    def score_oneVSone(self, X,y):
        score = sum(self.predict_oneVSone(X) == y) / len(y)
        return score


scaler = StandardScaler()
X_fit = scaler.fit_transform(X)
for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_fit, Y, test_size=0.33)
        log1 = LogisticRegression(n_iteration=30000).fit_oneVSone(X_train, y_train)
        pre = log1.predict_oneVSone(X_test)
        score1 = log1.score_oneVSone(X_test, y_test)

        logi = LogisticRegression(n_iteration=30000).fit_oneVSall(X_train, y_train)
        predition1 = logi.predict_oneVSall(X_test)
        score2 = logi.score_oneVSall(X_test, y_test)

        print("1vs1 acc",score1)
        print("1vsAll acc", score2)