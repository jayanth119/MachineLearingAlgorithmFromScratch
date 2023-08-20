import numpy as np

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label among the neighbors
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

# Create synthetic training data
X_train = np.random.rand(100, 5)  # 100 samples, 5 features
y_train = np.random.randint(0, 2, 100)  # Binary labels

# Create synthetic test data
X_test = np.random.rand(20, 5)  # 20 samples for testing

# Initialize and train the k-NN classifier
knn = KNNClassifier(k=3)
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)
print("Predictions:", predictions)
