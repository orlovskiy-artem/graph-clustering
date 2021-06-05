import numpy as np
from copy import deepcopy
import networkx as nx

from Tools import DisjointSet

EPS = 10e-5


class MCLClusterizer:
    def __init__(self,
                 power=2,
                 inflation_param=2,
                 N=100):
        self.A = None
        self.power = power
        self.inflation_param = inflation_param
        self.N = N
        self.clusters = None

    def _to_stockhastic(self):
        A_copy = deepcopy(self.A)
        self.A += np.eye(self.A.shape[0])
        for i in range(self.A.shape[0]):
            self.A[i] /= sum(A_copy[i])
        self.A = self.A.T

    def _inflate(self):
        self.A = self.A.T
        for i in range(self.A.shape[0]):
            self.A[i] = [*map(lambda x:x**self.inflation_param,
                              self.A[i])]
            self.A[i] /= sum(self.A[i])
        self.A = self.A.T

    def _mcl(self):
        mtx_base = deepcopy(self.A)
        for n in range(self.N):
            for p in range(self.power):
                self.A = self.A@mtx_base
            self._inflate()
            # self.A = deepcopy(self.A)

    def fit(self, graph):
        self.A = nx.to_numpy_array(graph)
        self._to_stockhastic()
        mtx_base = deepcopy(self.A)

        for n in range(self.N):
            for p in range(self.power):
                self.A = self.A@mtx_base
            self._inflate()
            mtx_base = deepcopy(self.A)
        return self

    def predict(self):
        self.clusters = DisjointSet()
        for i in range(self.A.shape[0]):
            self.clusters.create(i)

        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                if(abs(self.A[i][j]) > EPS):
                    self.clusters.union(i, j)

        return self.clusters.to_ls_of_sets()

    def fit_predict(self, graph):
        self.fit(graph)
        return self.predict()
