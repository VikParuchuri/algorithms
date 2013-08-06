from __future__ import division
from base import Algorithm, mean, Matrix
from copy import deepcopy

class LinregNonMatrix(Algorithm):
    """
    Solve linear regression with a single variable
    """
    def train(self, x, y):
        """
        x - a list of x values
        y - a list of y values
        """
        x_mean = mean(x)
        y_mean = mean(y)
        x_dev = sum([abs(i-x_mean) for i in x])
        y_dev = sum([abs(i-y_mean) for i in y])

        self.slope = (x_dev*y_dev)/(x_dev*x_dev)
        self.intercept = y_mean - (self.slope*x_mean)

    def predict(self, z):
        """
        z - a list of x values to predict on
        returns - computed y values for the input vector
        """
        return [i*self.slope + self.intercept for i in z]

class LinregCustom(Algorithm):
    """
    Solves for multivariate linear regression
    """
    def train(self, X, y):
        """
        X - input list of lists
        y - input column vector in list form, ie [[1],[2]]
        """
        assert len(y) == len(X)
        X_int = self.append_intercept(X)
        coefs = ((Matrix(X_int) * Matrix(X_int).transpose()).invert())
        coefs = (Matrix(X_int).transpose()) * coefs
        coefs = coefs * Matrix(y)
        self.coefs = coefs

    def predict(self,Z):
        """
        Z - input list of lists
        """
        Z = self.append_intercept(Z)
        return Matrix(Z) * self.coefs

    def append_intercept(self, X):
        """
        Adds the intercept term to the first row of a matrix
        """
        X = deepcopy(X)

        #Append this to calculate the intercept term properly
        for i in xrange(0,len(X)):
            X[i] = [1] + X[i]
        return X

class LinregNumpy(Algorithm):
    """
    Use numpy to solve a multivariate linear regression
    """
    def train(self,X,y):
        """
        X - input list of lists
        y - input column vector in list form, ie [[1],[2]]
        """
        from numpy import array,linalg, ones,vstack
        assert len(y) == len(X)
        X = vstack([array(X).T,ones(len(X))]).T
        self.coefs = linalg.lstsq(X,y)[0]
        self.coefs = self.coefs.reshape(self.coefs.shape[0],-1)

    def predict(self,Z):
        """
        Z - input list of lists
        """
        from numpy import array, ones,vstack
        Z = vstack([array(Z).T,ones(len(Z))]).T
        return Z.dot(self.coefs)

def fscore(rss1,rss2,p1,p2,N):
    """
    Use formula f = (rss1-rss2)(p2-p1)/(rss2)/(N-p2-1) to calculate f score.
    F score shows us how much residual error changes with each additional parameter in the "bigger" model, p1
    """
    diff = rss2-rss1
    numerator = diff * (p2-p1)
    denominator = rss1/(N-p2-1)
    return numerator/denominator

