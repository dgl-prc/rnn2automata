from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


class Partitioner(object):
    def fit(self, X):
        pass

    def predict(self, X):
        pass

    def get_fit_labels(self):
        pass


class Kmeans(Partitioner):

    def __init__(self, n_clusters):
        self.kmeans = KMeans(n_clusters=n_clusters)

    def fit(self, X):
        self.kmeans = self.kmeans.fit(X)

    def predict(self, X):
        return self.kmeans.predict(X)

    def get_fit_labels(self):
        return list(self.kmeans.labels_)


class PreditBasedKmeans(Partitioner):
    def __init__(self, n_clusters):
        self.kmeans = KMeans(n_clusters=n_clusters)


class EHCluster(Partitioner):
    '''
    Enhanced hierarchical clustering (one of the Agglomerative clustering).
    Originally, the Agglomerative clustering does not support prediction, but in our case, we need it.
    Thus, we introduce KNN to implement this.
    '''

    def __init__(self, n_clusters, n_neighbors=5):
        self.clustering = AgglomerativeClustering(n_clusters=n_clusters)
        self.neigh = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X):
        self.clustering.fit(X)
        self.neigh.fit(X, self.clustering.labels_)

    def predict(self, X):
        return self.neigh.predict(X)

    def get_fit_labels(self):
        return list(self.clustering.labels_)
