import numpy as np
from sklearn.cluster import KMeans as _KMeans, DBSCAN as _DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score


class KMeans:
    """K-Means clustering.

    Partitions data into k clusters by iteratively minimising the within-cluster
    sum of squared distances to the cluster centroid.

    Parameters
    ----------
    n_clusters : int, default=4
        Number of clusters to form.
    init : str, default="k-means++"
        Centroid initialisation strategy; "k-means++" spreads initial centroids
        to speed up convergence.
    max_iter : int, default=300
        Maximum number of EM iterations per run.
    **kwargs
        Passed to sklearn.cluster.KMeans.

    Example
    -------
    >>> model  = KMeans(n_clusters=4).train(X)
    >>> labels = model.predict(X)
    >>> report = model.evaluate(X)
    """
    def __init__(self, n_clusters=4, init="k-means++", max_iter=300, **kwargs):
        self.model = _KMeans(n_clusters=n_clusters, init=init,
                             max_iter=max_iter, random_state=42, **kwargs)

    def train(self, X):
        """Fit the model on unlabelled data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self.model.fit(X)
        return self

    def predict(self, X):
        """Assign each sample to its nearest centroid.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        return self.model.predict(X)

    def evaluate(self, X):
        """Compute clustering quality metrics on fitted data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The same data used for training.

        Returns
        -------
        dict with keys:
            inertia          — within-cluster sum of squared distances
            silhouette       — mean silhouette score (-1 worst, +1 best)
            davies_bouldin   — lower is better
            n_clusters_found — number of unique cluster labels
        """
        labels = self.model.labels_
        return {
            "inertia":          self.model.inertia_,
            "silhouette":       silhouette_score(X, labels),
            "davies_bouldin":   davies_bouldin_score(X, labels),
            "n_clusters_found": len(np.unique(labels)),
        }


class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Groups points that are closely packed together while marking points in
    low-density regions as noise (-1). Does not require specifying k upfront.

    Parameters
    ----------
    eps : float, default=0.5
        Maximum distance between two samples to be considered neighbours.
        Tune this to the scale of your data.
    min_samples : int, default=5
        Minimum number of samples in a neighbourhood to form a core point.
        Higher values → more points classified as noise.
    **kwargs
        Passed to sklearn.cluster.DBSCAN.

    Example
    -------
    >>> model  = DBSCAN(eps=0.3, min_samples=10).train(X)
    >>> labels = model.predict(X)
    >>> report = model.evaluate(X)
    """
    def __init__(self, eps=0.5, min_samples=5, **kwargs):
        self.model = _DBSCAN(eps=eps, min_samples=min_samples, **kwargs)

    def train(self, X):
        """Fit the model on unlabelled data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self.model.fit(X)
        return self

    def predict(self, X):
        """Return cluster labels from the fitted model.

        DBSCAN is transductive — it cannot assign labels to new unseen points.
        This returns the labels stored during train.

        Parameters
        ----------
        X : array-like (unused, kept for API consistency)

        Returns
        -------
        labels : ndarray of shape (n_samples,)  — -1 indicates noise
        """
        return self.model.labels_

    def evaluate(self, X):
        """Compute clustering quality metrics, excluding noise points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        dict with keys:
            n_clusters    — number of clusters found (excluding noise)
            n_noise       — number of samples labelled as noise (-1)
            silhouette    — mean silhouette score (only when ≥ 2 clusters)
            davies_bouldin — lower is better (only when ≥ 2 clusters)
        """
        labels = self.model.labels_
        mask = labels != -1
        if len(np.unique(labels[mask])) < 2:
            return {
                "n_clusters": len(np.unique(labels[mask])),
                "n_noise":    int((labels == -1).sum()),
            }
        return {
            "n_clusters":     len(np.unique(labels[mask])),
            "n_noise":        int((labels == -1).sum()),
            "silhouette":     silhouette_score(X[mask], labels[mask]),
            "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]),
        }


class Hierarchical:
    """Agglomerative (bottom-up) hierarchical clustering.

    Starts with each sample as its own cluster and repeatedly merges the
    two closest clusters until the target number of clusters is reached.

    Parameters
    ----------
    n_clusters : int, default=4
        Number of clusters to form.
    linkage : str, default="ward"
        Merge criterion: "ward" minimises within-cluster variance, "complete"
        uses the farthest pair, "average" uses the mean pairwise distance.
    **kwargs
        Passed to sklearn.cluster.AgglomerativeClustering.

    Example
    -------
    >>> model  = Hierarchical(n_clusters=4, linkage="ward").train(X)
    >>> labels = model.predict(X)
    >>> report = model.evaluate(X)
    """
    def __init__(self, n_clusters=4, linkage="ward", **kwargs):
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage, **kwargs)

    def train(self, X):
        """Fit the model on unlabelled data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self.model.fit(X)
        return self

    def predict(self, X):
        """Return cluster labels from the fitted model.

        Agglomerative clustering is transductive — labels are computed only
        for training samples. Returns the labels stored during train.

        Parameters
        ----------
        X : array-like (unused, kept for API consistency)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        return self.model.labels_

    def evaluate(self, X):
        """Compute clustering quality metrics on fitted data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        dict with keys:
            silhouette     — mean silhouette score (-1 worst, +1 best)
            davies_bouldin — lower is better
            n_clusters     — number of unique cluster labels
        """
        labels = self.model.labels_
        return {
            "silhouette":     silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "n_clusters":     len(np.unique(labels)),
        }


class GMM:
    """Gaussian Mixture Model clustering.

    Models data as a mixture of k Gaussian distributions. Each cluster has
    its own mean and covariance, making it more flexible than K-Means.
    Fitted via the Expectation-Maximisation (EM) algorithm.

    Parameters
    ----------
    n_components : int, default=4
        Number of Gaussian components (clusters).
    covariance_type : str, default="full"
        Covariance matrix constraint: "full" (each cluster has its own
        full matrix), "tied", "diag", or "spherical".
    max_iter : int, default=200
        Maximum number of EM iterations.
    **kwargs
        Passed to sklearn.mixture.GaussianMixture.

    Example
    -------
    >>> model  = GMM(n_components=4).train(X)
    >>> labels = model.predict(X)
    >>> report = model.evaluate(X)
    """
    def __init__(self, n_components=4, covariance_type="full",
                 max_iter=200, **kwargs):
        self.model = GaussianMixture(
            n_components=n_components, covariance_type=covariance_type,
            max_iter=max_iter, random_state=42, **kwargs)

    def train(self, X):
        """Fit the model on unlabelled data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self.model.fit(X)
        return self

    def predict(self, X):
        """Assign each sample to the most probable Gaussian component.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        return self.model.predict(X)

    def evaluate(self, X):
        """Compute clustering quality and model selection metrics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        dict with keys:
            aic            — Akaike Information Criterion (lower → better fit)
            bic            — Bayesian Information Criterion (lower → better fit, penalises complexity more)
            silhouette     — mean silhouette score
            davies_bouldin — lower is better
        """
        labels = self.model.predict(X)
        return {
            "aic":            self.model.aic(X),
            "bic":            self.model.bic(X),
            "silhouette":     silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
        }
