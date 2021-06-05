import random
from collections import defaultdict
from copy import deepcopy
import time
import networkx as nx
import numpy as np
from typing import Dict


class KMedoidsClusterizer:
    def __init__(self, n_clusters=3, random_state=time.time()):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.medoids = None
        # node2medoid  - means node_index2medoid_index, and the next is reverse dict
        self.node2medoid = None
        self.medoid2node = None
        # Q - modularity
        self.Q = None

    def fit(self, graph):
        """Fits by KMedoids, using jaccard as metrics and graph modularity as 
        loss function 

        Arguments:
            graph {nx.Graph} -- given graph

        Returns:
            List[Sets[Int]] -- list of sets with int as node_indices and sets as
                                clusters 

        A - Adjacency matrix
        k - degrees of nodes
        m - number of edges

        graph modularity formula check in 
        """

        A = nx.to_numpy_array(graph)
        k = sum(A)
        m = sum(k)/2
        random.seed(self.random_state)

        node_indices = [*range(A.shape[0])]
        medoids_indices = random.sample(range(A.shape[0]), self.n_clusters)
        medoid2nodes = defaultdict(list)
        node2medoid = {}

        # argmax by jaccard -- get best medoid for given node
        def argmax(node_index, medoid_indices): return max(map(lambda
                                                               medoid_index: (medoid_index, compute_jaccard(A, k,                                             medoid_index, node_index)), medoid_indices),
                                                           key=lambda item: item[1])[0]

        # 1) init medoids
        for node_index in node_indices:
            if node_index in medoids_indices:
                medoid2nodes[node_index].append(node_index)
                node2medoid[node_index] = node_index
            else:
                best_medoid_index = argmax(node_index, medoids_indices)
                medoid2nodes[best_medoid_index].append(node_index)
                node2medoid[node_index] = best_medoid_index

        # 2) get first modularity
        Q = compute_modularity(A, m, k, node2medoid)

        # 3) update medoids
        modularity_grows = True
        while modularity_grows:
            for medoid_index, indices_of_nodes_in_cluster in medoid2nodes.items():
                medoid_loop_flag = True
                for index_of_node_in_cluster in indices_of_nodes_in_cluster:
                    if index_of_node_in_cluster == medoid_index:
                        continue
                    # move medoid
                    new_medoid_index = index_of_node_in_cluster
                    # init new medoids
                    new_medoids_indices = deepcopy(medoids_indices)
                    new_medoids_indices.remove(medoid_index)
                    new_medoids_indices.append(new_medoid_index)
                    # init new partitioning
                    new_medoid2nodes = defaultdict(list)
                    new_node2medoid = {}
                    for node_index in node_indices:
                        if node_index in new_medoids_indices:
                            new_medoid2nodes[node_index].append(node_index)
                            new_node2medoid[node_index] = node_index
                        else:
                            best_medoid_index = argmax(
                                node_index, new_medoids_indices)
                            new_medoid2nodes[best_medoid_index].append(
                                node_index)
                            new_node2medoid[node_index] = best_medoid_index
                    Q_new = compute_modularity(A, m, k, new_node2medoid)
                    if Q_new > Q:
                        Q = Q_new
                        medoid2nodes = deepcopy(new_medoid2nodes)
                        medoids_indices = deepcopy(new_medoids_indices)
                        node2medoid = deepcopy(new_node2medoid)
                        medoid_loop_flag = False
                        break
                if not medoid_loop_flag:
                    break
                else:
                    modularity_grows = False
                    break
        self.Q = Q
        self.node2medoid = node2medoid
        self.medoid2node = medoid2nodes
        self.medoids = medoids_indices
        return self

    def predict(self):
        clusters2nodes = defaultdict(set)
        for node_index, cluster_index in self.node2medoid.items():
            clusters2nodes[cluster_index].add(node_index)
        return list(clusters2nodes.values())

    def fit_predict(self, graph: nx.Graph):
        self.fit(graph)
        return self.predict()


def compute_jaccard(adjacency_matrix: np.array,
                    node_degrees: np.array,
                    node_index_i: int,
                    node_index_j: int):

    adj_mtx = adjacency_matrix
    i = node_index_i
    j = node_index_j
    try:
        return adj_mtx[i][j]/(node_degrees[i]+node_degrees[j]-adj_mtx[i][j])
    except ZeroDivisionError:
        raise ZeroDivisionError(
            "Jaccard similarity is not proper for this matrix")


def compute_modularity(adjacency_matrix: np.array,
                       number_of_all_edges: int,
                       node_degrees: np.array,
                       node2cluster: Dict[int, int]):
    Q = 0
    A = adjacency_matrix
    k = node_degrees
    s = node2cluster
    m = number_of_all_edges

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Q += A[i][j]-k[i]*k[j]/(2*m)*(s[i] == s[j])
    Q /= 2*m
    return Q
