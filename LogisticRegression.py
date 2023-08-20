class Logistic_Regression:
    def __init__(self,learning_rate,no_of_iteration):
        self.learning_rate = learning_rate
        self.no_of_iteration = no_of_iteration
    def fit(self , x, y):
        self.m , self.n = x.shape
        self.x = x
        self.y = y 
        self.weights = np.zeros(self.n)
        self.bias = 0
        for i in range(no_of_iteration):
            self.process()
    def process():
        #z= w*x +b 
        z  = self.x.dot(self.weights)+self.bias
        y_dash = 1/(1+np.exp(-z)) #sigmoid_function 
        dw = (1/self.m)*np.dot(self.x.transpose(), y_dash-self.y)
        db = (1/self.m)*np.sum(y_dash-self.y)
        self.weights= self.weights - learning_rate*dw
        self.bias = self.bias-learning_rate*db
    def predict(self,x):
        z  = self.x.dot(self.weights)+self.bias
        y_dash = 1/(1+np.exp(-z)) #sigmoid_function
        result = np.where(y_dash>0.5,1,0)
        return result
