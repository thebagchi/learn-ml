import numpy as np
import umap as _umap
from sklearn.decomposition import PCA as _PCA
from sklearn.manifold import TSNE as _TSNE, trustworthiness


class PCA:
    """Principal Component Analysis for linear dimensionality reduction.

    Projects data onto the directions of maximum variance (principal components).
    Allows reconstruction of the original data, enabling a measurable
    reconstruction error.

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components to keep.
    **kwargs
        Passed to sklearn.decomposition.PCA.

    Example
    -------
    >>> model     = PCA(n_components=2).train(X)
    >>> X_reduced = model.predict(X)
    >>> report    = model.evaluate(X)
    """
    def __init__(self, n_components=2, **kwargs):
        self.model = _PCA(n_components=n_components, **kwargs)

    def train(self, X):
        """Fit PCA on the data (compute principal components).

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
        """Project data onto the principal components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_reduced : ndarray of shape (n_samples, n_components)
        """
        return self.model.transform(X)

    def evaluate(self, X):
        """Compute variance explained and reconstruction quality.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        dict with keys:
            explained_variance_ratio — per-component variance fraction
            total_explained_variance — sum of the above
            reconstruction_error_mse — mean squared error when projecting
                                       back to the original space
        """
        X_reconstructed = self.model.inverse_transform(self.model.transform(X))
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)
        return {
            "explained_variance_ratio": self.model.explained_variance_ratio_.tolist(),
            "total_explained_variance": float(self.model.explained_variance_ratio_.sum()),
            "reconstruction_error_mse": float(reconstruction_error),
        }


class TSNE:
    """t-SNE (t-distributed Stochastic Neighbour Embedding) for non-linear reduction.

    Minimises the KL divergence between pairwise similarity distributions in the
    original and low-dimensional spaces, preserving local neighbourhood structure.

    Note: t-SNE fits and transforms in a single pass (`fit_transform`). Calling
    `predict` after `train` returns the stored embedding; it cannot project new
    unseen points.

    Parameters
    ----------
    n_components : int, default=2
        Dimensionality of the embedding (2 or 3 for visualisation).
    perplexity : float, default=30
        Controls the effective number of neighbours considered.
        Typical range: 5–50. Larger datasets benefit from higher values.
    learning_rate : float or "auto", default=200
        Step size for the gradient descent optimisation.
    **kwargs
        Passed to sklearn.manifold.TSNE.

    Example
    -------
    >>> model      = TSNE(n_components=2, perplexity=30).train(X)
    >>> X_embedded = model.predict(X)
    >>> report     = model.evaluate(X_embedded)
    """
    def __init__(self, n_components=2, perplexity=30, learning_rate=200, **kwargs):
        self.model = _TSNE(n_components=n_components, perplexity=perplexity,
                           learning_rate=learning_rate, random_state=42, **kwargs)
        self._embedding = None

    def train(self, X):
        """Compute the t-SNE embedding (fit and transform in one step).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        # t-SNE has no separate fit/transform — fit_transform does both
        self._embedding = self.model.fit_transform(X)
        return self

    def predict(self, X):
        """Return the embedding computed during train.

        Parameters
        ----------
        X : array-like (unused — t-SNE cannot project new points)

        Returns
        -------
        X_embedded : ndarray of shape (n_samples, n_components)
        """
        return self._embedding

    def evaluate(self, X_embedded):
        """Report optimisation diagnostics from the fitted t-SNE.

        Parameters
        ----------
        X_embedded : ndarray of shape (n_samples, n_components)
            The embedding returned by predict.

        Returns
        -------
        dict with keys:
            kl_divergence  — final KL divergence (lower → better embedding)
            output_shape   — shape of the embedding array
            n_iter_actual  — number of optimisation iterations run
        """
        return {
            "kl_divergence": float(self.model.kl_divergence_),
            "output_shape":  list(X_embedded.shape),
            "n_iter_actual": int(self.model.n_iter_),
        }


class UMAP:
    """UMAP (Uniform Manifold Approximation and Projection) for non-linear reduction.

    Constructs a fuzzy topological representation of the data and optimises a
    low-dimensional layout that preserves both local and global structure better
    than t-SNE, while also supporting projection of new points.

    Requires the `umap-learn` package: pip install umap-learn

    Parameters
    ----------
    n_components : int, default=2
        Dimensionality of the embedding.
    n_neighbors : int, default=15
        Size of the local neighbourhood used for manifold approximation.
        Smaller values → more local detail; larger values → more global structure.
    min_dist : float, default=0.1
        Minimum distance between embedded points. Smaller values → tighter
        clusters; larger values → more uniform spread.
    **kwargs
        Passed to umap.UMAP.

    Example
    -------
    >>> model      = UMAP(n_components=2).train(X)
    >>> X_embedded = model.predict(X)
    >>> report     = model.evaluate(X, X_embedded)
    """
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, **kwargs):
        self.model = _umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                                min_dist=min_dist, random_state=42, **kwargs)

    def train(self, X):
        """Fit the UMAP model on the data.

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
        """Project data into the learned low-dimensional space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_embedded : ndarray of shape (n_samples, n_components)
        """
        return self.model.transform(X)

    def evaluate(self, X_original, X_embedded, n_neighbors=15):
        """Measure how well the embedding preserves local neighbourhood structure.

        Parameters
        ----------
        X_original : array-like of shape (n_samples, n_features)
            Original high-dimensional data.
        X_embedded : array-like of shape (n_samples, n_components)
            Low-dimensional embedding (returned by predict).
        n_neighbors : int, default=15
            Number of neighbours used in the trustworthiness calculation.

        Returns
        -------
        dict with keys:
            trustworthiness — fraction of k nearest neighbours in the embedding
                              that were also neighbours in the original space (0–1)
            output_shape    — shape of the embedding array
        """
        trust = trustworthiness(X_original, X_embedded, n_neighbors=n_neighbors)
        return {
            "trustworthiness": float(trust),
            "output_shape":    list(X_embedded.shape),
        }
