from __future__ import division
from copy import deepcopy


class Algorithm(object):
    def __init__(self):
        pass

    def train(self, X, y):
        pass

    def predict(self, X, y):
        pass

class Matrix(object):
    def __init__(self, X):
        self.validate(X)
        self.X = X

    def validate(self, X):
        list_error = "X must be a list of lists corresponding to a matrix, with each sub-list being a row."
        if not isinstance(X, list):
            raise Exception(list_error)
        if not isinstance(X[0], list):
            raise Exception(list_error)
        first_row_len = len(X[0])
        for i in xrange(0,len(X)):
            if len(X[i])!=first_row_len:
                raise Exception("All rows in X must be the same length.")

    def invert(self):
        return invert(self.X)
    
    @property
    def rows(self):
        return len(self.X)
    
    @property
    def cols(self):
        return len(self.X[0])
    
    def transpose(self):
        trans = []
        for j in xrange(0,self.cols):
            row = []
            for i in xrange(0,self.rows):
                row.append(self.X[i][j])
            trans.append(row)
        self.X = trans

    def __mul__(self, Z):
        self.validate(Z)

        z_rows = len(Z)
        z_cols = len(Z[0])
        assert z_rows==self.cols

        product = []
        for i in xrange(0,self.rows):
            row = []
            for j in xrange(0,z_cols):
                row.append(row_multiply(self.X[i], [Z[m][j] for m in xrange(0,z_rows)]))
            product.append(row)
        return product

def row_multiply(r1,r2):
    assert(len(r1)==len(r2))
    products =[]
    for i in xrange(0,len(r1)):
        products.append(r1[i]*r2[i])
    return sum(products)

def check_for_all_zeros(X,i,j):
    non_zeros = []
    first_non_zero = -1
    for m in xrange(i,len(X)):
        non_zero = X[m][j]!=0
        non_zeros.append(non_zero)
        if first_non_zero==-1 and non_zero:
            first_non_zero = m
    zero_sum = sum(non_zeros)
    return zero_sum, first_non_zero

def swap_row(X,i,p):
    X[p], X[i] = X[i], X[p]
    return X

def make_identity(r,c):
    identity = []
    for i in xrange(0,r):
        row = []
        for j in xrange(0,c):
            elem = 0
            if i==j:
                elem = 1
            row.append(elem)
        identity.append(row)
    return identity

def invert(X):
    """
    Invert a matrix X according to gauss-jordan elimination
    In gauss-jordan elimination, we perform basic row operations to turn a matrix into
    row-echelon form.  If we concatenate an identity matrix to our input
    matrix during this process, we will turn the identity matrix into our inverse.
    X - input list of lists where each list is a matrix row
    output - inverse of X
    """
    #copy X to avoid altering input
    X = deepcopy(X)

    #Get dimensions of X
    rows = len(X)
    cols = len(X[0])

    #Get the identity matrix and append it to the right of X
    #This is done because our row operations will make the identity into the inverse
    identity = make_identity(rows,cols)
    for i in xrange(0,rows):
        X[i]+=identity[i]

    i = 0
    for j in xrange(0,cols):
        print("On col {0} and row {1}".format(j,i))
        #Check to see if there are any nonzero values below the current row in the current column
        zero_sum, first_non_zero = check_for_all_zeros(X,i,j)
        #If everything is zero, increment the columns
        if zero_sum==0:
            if j==cols:
                return X
            raise Exception("Matrix is singular.")
        #If X[i][j] is 0, and there is a nonzero value below it, swap the two rows
        if first_non_zero != i:
            X = swap_row(X,i,first_non_zero)
        #Divide X[i] by X[i][j] to make X[i][j] equal 1
        X[i] = [m/X[i][j] for m in X[i]]

        #Rescale all other rows to make their values 0 below X[i][j]
        for q in xrange(0,rows):
            if q!=i:
                scaled_row = [X[q][j] * m for m in X[i]]
                X[q]= [X[q][m] - scaled_row[m] for m in xrange(0,len(scaled_row))]
        #If either of these is true, we have iterated through the matrix, and are done
        if i==rows or j==cols:
            break
        i+=1

    #Get just the right hand matrix, which is now our inverse
    for i in xrange(0,rows):
        X[i] = X[i][cols:len(X[i])]

    return X

def mean(l):
    sum(l)/len(l)

def gje(X):
    X = deepcopy(X)
    rows = len(X)
    cols = len(X[0])

    identity = make_identity(rows,cols)
    for i in xrange(0,rows):
        X[i]+=identity[i]

    for k in xrange(0,rows):
        abs_list = [abs(X[i][k]) for i in xrange(k,rows)]
        i_max  = abs_list.index(max(abs_list))+k
        if X[i_max][k] == 0:
            raise Exception("Matrix is singular!")
        X = swap_row(X, k, i_max)
        for i in xrange(k+1, rows):
            for j in xrange(k, cols):
                X[i][j]  = X[i][j] - X[k][j] * (X[i][j] / X[j][j])
            X[i][k]  = 0

    for i in xrange(0,rows):
        X[i] = X[i][int(cols):len(X[i])]

    return X
