from copy import deepcopy
from collections import defaultdict
import numpy as np
import numpy.linalg as LA
import networkx as nx
from sklearn.cluster import KMeans

from Tools import DisjointSet


class SpectralEmbeddingEstimator:
    def __init__(self):
        self.graph = None
        self.laplacian = None
        self.eigen_vectors = None
        self.eigen_values = None

    def fit(self, graph):
        self.graph = deepcopy(graph)
        t = np.array([*map(int,np.array(list(self.graph.degree())).T[1].T)])
        Laplacian = np.eye(t.shape[0])*t - nx.to_numpy_array(self.graph)
        self.laplacian = Laplacian
        self.eigenvalues, self.eigenvectors = LA.eig(self.laplacian)

    def predict(self, dimension=2):
        if dimension > self.laplacian.shape[0] - 1 or dimension < 1:
            raise ValueError(f"Wrong dimension: {dimension}")
        
        spectral_embeds = np.array([*map(lambda item: item[1],
                                         sorted(dict(zip(self.eigenvalues,
                                                         self.eigenvectors.T)).items(),
                                                key=lambda item: item[0]))][1:dimension+1]).T
        
        return spectral_embeds

    def fit_predict(self, graph, dimension=2):
        self.fit(graph)
        return self.predict(dimension)

    # def _get_laplacian(self):
    # def get_graph_node_embs(self, dimension = 2):
    #     self._get_laplacian()


# доп идея - модельку можно переделать на лад скилерна или
# сделать аггрегацию и наоборот
# основная идея - не отлавливать ошибки во время принимания параметров
# в диалговом окне, а отлавливать конфигуратором
# таким образом передача ошибки будет идти наверх в юай, где поднимем окошко
# что все пошло наперекосяк


# TODO
# тудушки - дедлайн - 2 ДНЯ до 27, 27 пишем доки
# 0) поправить идею выше
# 1) подключить марковский кластеризатор
# 2) подключить Медоидовский кластеризатор
# 3) Добавить редакцию графа
# 4) Добавить поддержку файлов
# 5) Добавить дублирование вкладок (опционально, но удобно) ===== YES
# 6) ...

class SpectralClusterizer(KMeans):
    def __init__(self, n_clusters,**kwargs):
        super().__init__(n_clusters,**kwargs)

    def fit(self, node_embeds):
        super().fit(node_embeds)

    def fit_predict(self, node_embeds):
        self.fit(node_embeds)
        return self.predict(node_embeds)

    def predict(self, node_embeds):
        nodes_clusters_indices = super().predict(node_embeds)
        clusters_indices2sets = defaultdict(set)
        for node_index,cluster_index in enumerate(nodes_clusters_indices):
            clusters_indices2sets[cluster_index].add(node_index)
        return list(clusters_indices2sets.values())


class SpectralConfigurator:
    def __init__(self, configs):
        self.node_embed_dimension = configs.get("node_embedding_dimension", 2)
        self.n_clusters = configs.get("n_clusters", 3)
        self.init = configs.get("init", 'k-means++')
        self.n_init = configs.get("n_init", 10)
        self.max_iter = configs.get("max_iter", 300)
        self.tol = configs.get("tol", 0.0001)
        # self.precompute_distances = configs.get(
        #     "precompute_distances", 'deprecated')
        self.verbose = configs.get("verbose", 0)
        self.random_state = configs.get("random_state", None)
        self.copy_x = configs.get("copy_x", True)
        # self.n_jobs = configs.get("n_jobs", 'deprecated')
        self.algorithm = configs.get("algorithm", "auto")

    def get_params_estimator(self):
        return {"dimension":self.node_embed_dimension}

    def get_params_clusterizer(self):
        return {"n_clusters" : self.n_clusters,
            "init":self.init,
            "n_init":self.n_init,
            "max_iter": self.max_iter,
            "tol":self.tol,
            "random_state":self.random_state,
            "copy_x":self.copy_x,
            "algorithm":self.algorithm}


# "n_jobs":self.n_jobs,
            
# "precompute_distances":self.precompute_distances,
            

# (self.n_clusters, self.init, self.n_init, self.max_iter, self.tol,\
#             self.precompute_distances, self.random_state, self.copy_x, self.n_jobs,\
#             self.algorithm)
            


# prev
# init='k-means++',
        # n_init=10,
        # max_iter=300,
        # tol=0.0001,
        # precompute_distances='deprecated',
        # verbose=0,
        # random_state=None,
        # copy_x=True,
        # n_jobs='deprecated',
        # algorithm='auto'
        # self.node_embed_dimension = node_embed_dimension
        # self.n_clusters =  n_clusters
        # self.init = init
        # self.n_init =  n_init
        # self.max_iter =  max_iter
        # self.tol =  tol
        # self.precompute_distances = precompute_distances
        # self.random_state = random_state
        # self.copy_x = copy_x
        # self.n_jobs = n_jobs
        # self.algorithm = algorithm
