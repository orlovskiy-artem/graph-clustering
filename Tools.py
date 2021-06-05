from collections import defaultdict


class DisjointSet:
    def __init__(self):
        self.parents = dict()

    def create(self, key):
        if key in self.parents:
            raise Exception("Key is already in Disjoint-Set")
        self.parents[key] = key

    def find(self, key):
        if key not in self.parents:
            return
        if key == self.parents[key]:
            return key
        return self.find(self.parents[key])

    def union(self, key_1, key_2):
        if key_1 not in self.parents or key_2 not in self.parents:
            return
        key_1 = self.find(key_1)
        key_2 = self.find(key_2)
        self.parents[key_1] = key_2

    def to_ls_of_sets(self):
        res = defaultdict(set)
        for k in self.parents:
            parent = self.find(k)
            res[parent].add(k)
        return list(res.values())

    def __len__(self):
        return len(self.parents)

    def __repr__(self):
        return f"DisjointSet({self.to_ls_of_sets()})"


def kmeans_pred2ls_of_sets(cluster_labels, node_index2node_label=None):
    clusters = defaultdict(set)
    if not node_index2node_label:
        for node_index, cluster_label in enumerate(cluster_labels):
            clusters[cluster_label].add(node_index)
    else:
        for node_index, cluster_label in enumerate(cluster_label):
            clusters[cluster_label].add(node_index2node_label[node_index])
    return list(clusters.values())
