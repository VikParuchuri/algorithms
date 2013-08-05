from __future__ import division
import math
from copy import deepcopy
from base import Algorithm, mean, Matrix

class KNNNonMatrix(Algorithm):
    """
    Solve knn without matrix
    """
    def train(self, X, y):
        """
        X - a list of x values
        y - a list of y values
        """
        self.X = X
        self.y = y

    def predict(self, z, neighbor_count=3):
        """
        z - a list of lists of x values to predict on
        returns - computed y values for the input list of lists
        """
        if neighbor_count>len(self.X):
            raise Exception("More neighbors specified than exist in training data.")
        x_len = len(self.X[0])
        predictions = []
        for i in xrange(0,len(z)):
            if len(z[i])!=x_len:
                raise Exception("Each input row must have the same length as each row in training data.")

            min_inds = find_min_indices(z[i],self.X,neighbor_count)
            y_values = []
            for m in min_inds:
                y_values.append(self.y[m])
            predictions.append(sum(y_values)/len(y_values))
        return predictions

def find_min_indices(v1,m,min_count):
    dists = []
    for row in m:
        dists.append(distance(v1,row))
    min_inds = []
    cdist = deepcopy(dists)
    for i in xrange(0,min_count):
        mdist = min(cdist)
        min_inds.append(dists.index(mdist))
        cdist.remove(mdist)
    return min_inds

def distance(v1,v2):
    if len(v1)!=len(v2):
        raise Exception("Lengths must match in order to calculate distance!")

    dists = []
    for i in xrange(0,len(v1)):
        dists.append((v1[i]-v2[1])**2)
    return math.sqrt(sum(dists))