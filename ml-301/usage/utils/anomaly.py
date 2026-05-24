import numpy as np
from sklearn.ensemble import IsolationForest as _IF
from sklearn.svm import OneClassSVM as _OCSVM
from sklearn.neighbors import LocalOutlierFactor as _LOF
from sklearn.metrics import precision_score, recall_score, roc_auc_score


class _AnomalyBase:
    def predict(self, X):
        """Predict anomaly labels for new samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            1 = anomaly, 0 = normal
        """
        # All detectors return -1 for anomaly — convert to 0/1
        raw = self.model.predict(X)
        return (raw == -1).astype(int)

    def evaluate(self, X_test, y_test):
        """Compute anomaly detection metrics on labelled test data.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
        y_test : array-like of shape (n_samples,)
            Ground-truth labels: 1 = anomaly, 0 = normal.

        Returns
        -------
        dict with keys:
            precision — of all flagged anomalies, fraction that are true anomalies
            recall    — of all true anomalies, fraction that were caught
            roc_auc   — area under the ROC curve using continuous anomaly scores
        """
        y_pred = self.predict(X_test)
        scores = -self.model.score_samples(X_test)
        return {
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "roc_auc":   roc_auc_score(y_test, scores),
        }


class IsolationForest(_AnomalyBase):
    """Isolation Forest anomaly detector.

    Isolates anomalies by randomly partitioning features. Anomalies are
    isolated in fewer splits than normal points, giving them a lower score.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of isolation trees; more trees → more stable scores.
    contamination : float, default=0.1
        Expected proportion of anomalies in the data. Used to set the
        decision threshold. Typical range: 0.01–0.2.
    **kwargs
        Passed to sklearn.ensemble.IsolationForest.

    Example
    -------
    >>> model  = IsolationForest(contamination=0.1).train(X_normal)
    >>> preds  = model.predict(X_test)   # 1 = anomaly, 0 = normal
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, n_estimators=100, contamination=0.1, **kwargs):
        self.model = _IF(n_estimators=n_estimators, contamination=contamination,
                         random_state=42, **kwargs)

    def train(self, X_normal):
        """Fit on normal (inlier) training data only.

        Parameters
        ----------
        X_normal : array-like of shape (n_samples, n_features)
            Training data containing only normal samples.

        Returns
        -------
        self
        """
        self.model.fit(X_normal)
        return self


class OneClassSVM(_AnomalyBase):
    """One-Class SVM anomaly detector.

    Learns a tight decision boundary around normal training data using a
    kernel function. Points outside the boundary are flagged as anomalies.

    Parameters
    ----------
    nu : float, default=0.1
        Upper bound on the fraction of training errors and lower bound on
        the fraction of support vectors. Roughly the expected contamination rate.
    kernel : str, default="rbf"
        Kernel type: "linear", "poly", "rbf", or "sigmoid".
    gamma : str or float, default="scale"
        Kernel coefficient; "scale" uses 1 / (n_features * X.var()).
    **kwargs
        Passed to sklearn.svm.OneClassSVM.

    Example
    -------
    >>> model  = OneClassSVM(nu=0.1).train(X_normal)
    >>> preds  = model.predict(X_test)   # 1 = anomaly, 0 = normal
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, nu=0.1, kernel="rbf", gamma="scale", **kwargs):
        self.model = _OCSVM(nu=nu, kernel=kernel, gamma=gamma, **kwargs)

    def train(self, X_normal):
        """Fit on normal (inlier) training data only.

        Parameters
        ----------
        X_normal : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self.model.fit(X_normal)
        return self


class LOF(_AnomalyBase):
    """Local Outlier Factor (novelty detection mode) anomaly detector.

    Compares the local density of a sample to the densities of its neighbours.
    Points in low-density regions relative to their neighbours get high outlier scores.

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbours used to compute local density.
        Smaller values make the detector more sensitive to local structure.
    contamination : float, default=0.1
        Expected proportion of anomalies; sets the decision threshold.
    **kwargs
        Passed to sklearn.neighbors.LocalOutlierFactor.

    Example
    -------
    >>> model  = LOF(n_neighbors=20, contamination=0.1).train(X_normal)
    >>> preds  = model.predict(X_test)   # 1 = anomaly, 0 = normal
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, n_neighbors=20, contamination=0.1, **kwargs):
        # novelty=True allows predict/score_samples on new data
        self.model = _LOF(n_neighbors=n_neighbors, contamination=contamination,
                          novelty=True, **kwargs)

    def train(self, X_normal):
        """Fit on normal (inlier) training data only.

        Parameters
        ----------
        X_normal : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self.model.fit(X_normal)
        return self
