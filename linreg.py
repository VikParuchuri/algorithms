from base import Algorithm, mean, Matrix

class LinregNonMatrix(Algorithm):
    def train(self, x, y):
        x_mean = mean(x)
        y_mean = mean(y)
        x_dev = sum([i-x_mean for i in x])
        y_dev = sum([i-y_mean for i in y])

        self.slope = (x_dev*y_dev)/(x_dev*x_dev)
        self.intercept = y_mean - (self.slope*x_mean)

    def predict(self, x):
        return [i*self.slope + self.intercept for i in x]

class LinregListMatrix(Algorithm):
    def train(self, X, y):
        X = Matrix(X)


