"""
Computational Complexity
The Normal Equation computes the inverse of X⊺ X, which is an (n + 1) × (n + 1)
matrix (where n is the number of features). The computational complexity of inverting
such a matrix is typically about O(n2.4) to O(n3), depending on the implementation. In
other words, if you double the number of features, you multiply the computation
time by roughly 22.4 = 5.3 to 23 = 8.
The SVD approach used by Scikit-Learn’s LinearRegression class is about O(n2). If
you double the number of features, you multiply the computation time by roughly 4.

Now we will look at a very different way to train a Linear Regression model, which is
better suited for cases where there are a large number of features or too many training
instances to fit in memory.
I bulid this linear regression model using  GradientDecent

An important parameter in Gradient Descent is the size of the steps, determined by
the learning rate hyperparameter.
Learning rate is used to change the parameter/weight of the data .
The intituation behind the learning rate is to give best weights from the error or costfunction
If the learning rate is too small, then the algorithm
will have to go through many iterations to converge, which will take a long time
On the other hand, if the learning rate is too high, you might jump across the valley
and end up on the other side, possibly even higher up than you were before. This
might make the algorithm diverge, with larger and larger values, failing to find a good
solution
****
When using Gradient Descent, you should ensure that all features
have a similar scale (e.g., using Scikit-Learn’s StandardScaler
class), or else it will take much longer to converge.
***
Linear Regression model when there are hun‐
dreds of thousands of features is much faster using Gradient
Descent than using the Normal Equation or SVD decomposition.
simulated annealing
generally in sklearn linearregression model we use least square regression using svd
linear regression can be done using
svd  in scipy there is an function called lstq()
gradient decent
batch gradient decent ---- it will use linear learning rate and the cost functin to find the weighths of linearregression 
mini  batch gradient decent ---- it will like batch gradient decent  of mini batch 
stochiastic gradient decent  --- it will use the learning rate by randomly and cost function of thr linear regression 
So when should you use plain Linear Regression (i.e., without any regularization),
Ridge, Lasso, or Elastic Net? It is almost always preferable to have at least a little bit of
regularization, so generally you should avoid plain Linear Regression. Ridge is a good
default, but if you suspect that only a few features are useful, you should prefer Lasso
or Elastic Net because they tend to reduce the useless features’ weights down to zero,
as we have discussed.
"""

import numpy as np
import pandas as pd 
class Linear_Regression:
    def __init__(self,learning_rate,no_of_iteration):
        self.learning_rate = learning_rate
        self.no_of_iteration = no_of_iteration
    def fit(self,x,y):
        print(x.shape)
        self.m , self.n =x.shape
        self.w  = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y
        for i in range(self.no_of_iteration):
            self.Gradient_Decent()
    def Gradient_Decent(self):
        y_predict = self.predict(self.x)
        dw =-(2*np.dot(self.x.T,self.y-y_predict))/self.m #it is covexfunction which means that if you pick any two points on the curve, the linesegment joining them never crosses the curve.
        db = -2*np.sum(self.y-y_predict)/self.m
        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db
    def predict(self,x):
        result = np.dot(x,self.w)+self.b
        return result
    def Weight_Bias():
        return np.array([self.w,self.b])
    def Regression(self,xdata,ydata):
        self.xdata = xdata
        self.ydata = ydata
        
        n = np.array(xdata).transpose()
        n = np.dot(n,xdata)
        print(xdata.shape)
        print(n.shape)
        s = np.array(np.transpose(xdata))
        s = np.dot(s,ydata)
        print(s.shape)
        print("xxxxx")
        x = np.dot(s,np.linalg.inv(n))
        print(x.shape)
        return x
