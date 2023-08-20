import numpy as np
class PCA:
    def __init__(self , n_components ):
        self.n_components = n_components
    def fit_transform(self , x ):
        #find covariance
        covmatrix = np.cov(x , rowvar = False )
        eigval,eigvect = np.linalg.eig(covmatrix)
        sorted_indices = np.argsort(eigval)[::-1]
        sorted_eigenvalues = eigval[sorted_indices]
        sorted_eigenvect = eigvect[:, sorted_indices]
        projection_matrix = sorted_eigenvect[:, :self.n_components]
        transformed_data = np.dot(x, projection_matrix)
        return transformed_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
pca = PCA(1)
x =  np.array([[1,2 ] , [3,3] ,[3,5],[5,4] , [5,6]  , [6,5] , [8,7]  , [9,8]])
print(x.shape)
x = pca.fit_transform(x)
plt.scatter(x , np.zeros(x.shape))
plt.scatter([1,3,3,5,5,6,8,9] ,[2,3,5,4,6,5,7,8])
plt.show()
print(x)
fig = plt.figure()
x =  np.array([[1,2 ,3] , [3,3 ,4 ] ,[3,5 ,6 ],[5,4 , 5] , [5,6 , 7 ]  , [6,5 , 6 ] , [8,7 , 8]  , [9,8 , 9 ]])
ax = fig.add_subplot(111, projection='3d')
ax.plot([1,3,3,5,5,6,8,9] ,[2,3,5,4,6,5,7,8] , [3,4,6,5,7,6,8,9])
plt.show()
