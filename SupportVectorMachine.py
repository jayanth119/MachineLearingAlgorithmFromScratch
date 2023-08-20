"""
The LinearSVC class is based on the liblinear library, which implements an opti‐
mized algorithm for linear SVMs.1 It does not support the kernel trick, but it scales
almost linearly with the number of training instances and the number of features. Its
training time complexity is roughly O(m × n).
The algorithm takes longer if you require very high precision.
The SVC class is based on the libsvm library, which implements an algorithm that
supports the kernel trick.2 The training time complexity is usually between O(m2 × n)
and O(m3 × n).
"""

import numpy as np 
class SVM():
    def __init__(self,learning_rate,no_of_iteration,lambda_parameter):
        self.learning_rate = learning_rate
        self.no_of_iteration = no_of_iteration
        self.lambda_parameter = lambda_parameter
    def fit(self,x,y):
        self.m , self.n = x.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y  = y
        for i in range(self.no_of_iteration):
            self.update_weight() #changes/update  the weights by using gradient descent 
    def update_weight(self):
        y_label = np.where(self.y<=0,-1,1) #converting  0 to -1 and 1 to 1 
        for  i,j in enumerate(self.x):
            #Gradient descent 
            if(y_label[i]*(np.dot(j,self.w)-self.b)>=1):
               dw = 2*self.lambda_parameter*self.w 
               db = 0
            else :
                dw = 2*self.lambda_parameter*self.w-np.dot(j,y_label[i])
                db = y_label[i]
        self.w = self.w-self.learning_rate*dw
        self.b = self.b-self.learning_rate*db
    def predict(self,x):
        output = np.dot(x,self.w)-self.b
        re= np.sign(output)
        result =  np.where(re<=-1,0,1)
        return result 
print("ok done " )
model = SVM(learning_rate=0.001,no_of_iteration=1000,lambda_parameter=0.01)
