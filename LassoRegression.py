import numpy as np 
class Lasso_Regression :
    def __init__(self,learning_rate ,no_of_iteration , penalty_term):
        self.learning_rate = learning_rate
        self.no_of_iteration = no_of_iteration
        self.penalty_term = penalty_term
    def fit(self,x,y):
        self.m  , self.n = x.shape
        self.x = x
        self.y = y
        self.w = np.zeros(self.n)
        self.b = 0
        for i in range(self.no_of_iteration):
            self.process()
            
        
    def process(self):
        y_dash = self.predict(self.x)
        dw = np.zeros(self.n)
        for  i in range(self.n):
            if(self.w > 0 ):
                dw[i] = (-2*(self.x[:i]).dot(self.y-y_dash)+self.panalty_term)/self.m
            else :
                dw[i] = (-2*(self.x[:i]).dot(self.y-y_dash) - self.panalty_term)/self.m
        db = -2*np.sum(self.y-y_dash)/self.m
        self.w = self.w- self.learning_rate*dw
        self.b = self.b - self.learning_rate*db
        
    def  predict(self,x):
        return  x.dot(self.w)+self.b
        
