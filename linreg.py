from base import Algorithm, mean, Matrix
from copy import deepcopy
from numpy import array,linalg, ones,vstack

class LinregNonMatrix(Algorithm):
    def train(self, x, y):
        x_mean = mean(x)
        y_mean = mean(y)
        x_dev = sum([i-x_mean for i in x])
        y_dev = sum([i-y_mean for i in y])

        self.slope = (x_dev*y_dev)/(x_dev*x_dev)
        self.intercept = y_mean - (self.slope*x_mean)

    def predict(self, Z):
        return [i*self.slope + self.intercept for i in Z]

class LinregListMatrix(Algorithm):
    def train(self, X, y):
        X_int = self.append_intercept(X)
        coefs = ((Matrix(X_int) * Matrix(X_int).transpose()).invert())
        coefs = (Matrix(X_int).transpose()) * coefs
        coefs = coefs * Matrix(y)
        self.coefs = coefs

    def predict(self,Z):
        Z = self.append_intercept(Z)
        return Matrix(Z) * self.coefs

    def append_intercept(self, X):
        X = deepcopy(X)

        #Append this to calculate the intercept term properly
        for i in xrange(0,len(X)):
            X[i] = [1] + X[i]
        return X

class LinregNumpy(Algorithm):
    def train(self,X,y):
        X = vstack([array(X).T,ones(len(X))]).T
        self.coefs = linalg.lstsq(X,y)[0]
        self.coefs = self.coefs.reshape(self.coefs.shape[0],-1)

    def predict(self,Z):
        Z = vstack([array(Z).T,ones(len(Z))]).T
        return Z.dot(self.coefs)





