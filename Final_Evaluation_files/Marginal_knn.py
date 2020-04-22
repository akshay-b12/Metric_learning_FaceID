import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class Marginal_knn:
    def __init__(self, n_neighbors):
        self.k = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.knn.fit(X,y)
    
    def predict(self,X):
        x_neigh_dist, x_neigh_ind = self.knn.kneighbors(X, n_neighbors=self.k, return_distance=True)
        classes = self.knn.classes_
        probs = []
        for i in range(len(X)-1):
            ni = np.zeros(len(classes), dtype=int)
            for p in x_neigh_ind[i]:
                ni[self.y[p]]+=1
            for j in range(i+1,len(X)):
                nj = np.zeros(len(classes), dtype=int)
                for p in x_neigh_ind[j]:
                    nj[self.y[p]]+=1
                probs.append(np.sum(ni*nj)/(self.k*self.k))
        return probs
    '''
    def predict_proba(self, X):
        ip_neigh_dist, ip_neigh_ind = self.knn.kneighbors(self.X, n_neighbors=self.k, return_distance=True)
        x_neigh_dist, x_neigh_ind = self.knn.kneighbors(X, n_neighbors=self.k, return_distance=True)
        class_probs = np.zeros((len(X), len(y)), dtype = float)
        classes = knn.classes_
        for i in len(X):  ## for all samples xi
            ni = np.zeros(len(classes), dtype=int)
            for p in x_neigh_ind[i]:
                ni[y[p]]+=1
            for j in len(self.k):  ## for all neighbours xj of xi
                idx = x_neigh_ind[i,j]
                nj = np.zeros(len(classes), dtype=int)
                for p in ip_neigh_ind[idx]:
                    nj[y[p]]+=1
                class_probs[y[idx]] = np.sum(ni*nj)/(self.k*self.k)
        return class_probs
    def score(self, X, y):
        probs = self.predict_proba(X)
        pred_class = []
        for i in len(X):
            pred_class = 
    '''