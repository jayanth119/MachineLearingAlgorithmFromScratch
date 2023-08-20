from  sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import random as rn 
class Kmeans:
    def __init__(self,k_clusters=2  , max_itr = 100):
        self.k = k_clusters
        self.max_itr = max_itr
        self.centriod = None
    def fit_predict(self , x):
        vr = rn.sample(range(0,x.shape(0)),2)
        self.centriod = x[vr]
        for i in range(self.max_itr):
            cluster_group = self.assign_clus(x) # assigning in clustergroup 
            old_cen =  self.centriod #preserve old centriod 
            self.centriod = self.change(x , cluster_group) #update the cluster group 
            if(self.centriod == old_cen).all():
                break
        return cluster_group            
    def assign_clus(self, x):
        clus_group = []
        dis = []
        for i in x :
            for j in  self.centriod :
                dis.append(np.sqrt(np.dot(i-j, i-j)))
        min_dis = min(dis)
        pos = np.array(dist.index(min_dis))
        return pos
    def change(self , x , cluster_group):
         new_cen = []
         for i in np.unique(cluster_group):
             new_cen.append(x[cluster_group==i].mean(axis = 0 ))
         return new_cen
             
        
                
        
