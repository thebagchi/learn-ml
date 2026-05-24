from sklearn.linear_model import LogisticRegression as _LR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix)
from xgboost import XGBClassifier


class _ClassificationBase:
    def train(self, X_train, y_train):
        """Fit the model on training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
        y_train : array-like of shape (n_samples,)  — binary or multi-class labels

        Returns
        -------
        self  (enables chaining: Model().train(X, y).predict(X_test))
        """
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Compute classification metrics on test data.

        Parameters
        ----------
        X_test  : array-like of shape (n_samples, n_features)
        y_test  : array-like of shape (n_samples,)

        Returns
        -------
        dict with keys:
            accuracy, precision, recall, f1, roc_auc, confusion_matrix
        """
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        return {
            "accuracy":         accuracy_score(y_test, y_pred),
            "precision":        precision_score(y_test, y_pred, zero_division=0),
            "recall":           recall_score(y_test, y_pred, zero_division=0),
            "f1":               f1_score(y_test, y_pred, zero_division=0),
            "roc_auc":          roc_auc_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }


class LogisticRegression(_ClassificationBase):
    """Logistic regression classifier.

    Parameters
    ----------
    C : float, default=1.0
        Inverse regularisation strength; smaller → stronger regularisation.
    max_iter : int, default=1000
        Maximum iterations for the solver to converge.
    **kwargs
        Passed to sklearn.linear_model.LogisticRegression.

    Example
    -------
    >>> model  = LogisticRegression(C=1.0).train(X_train, y_train)
    >>> preds  = model.predict(X_test)
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, C=1.0, max_iter=1_000, **kwargs):
        self.model = _LR(C=C, max_iter=max_iter, **kwargs)


class DecisionTree(_ClassificationBase):
    """Decision tree classifier.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree; None grows until leaves are pure.
    min_samples_split : int, default=2
        Minimum samples required to split an internal node.
    **kwargs
        Passed to sklearn.tree.DecisionTreeClassifier.

    Example
    -------
    >>> model  = DecisionTree(max_depth=5).train(X_train, y_train)
    >>> preds  = model.predict(X_test)
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, max_depth=None, min_samples_split=2, **kwargs):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_split=min_samples_split,
            random_state=42, **kwargs)


class RandomForest(_ClassificationBase):
    """Random forest classifier (bagged ensemble of decision trees).

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of each tree; None grows until leaves are pure.
    **kwargs
        Passed to sklearn.ensemble.RandomForestClassifier.

    Example
    -------
    >>> model  = RandomForest(n_estimators=200).train(X_train, y_train)
    >>> preds  = model.predict(X_test)
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, n_estimators=100, max_depth=None, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=42, **kwargs)


class SVM(_ClassificationBase):
    """Support vector machine classifier (kernel SVM).

    Parameters
    ----------
    C : float, default=1.0
        Regularisation parameter; smaller → wider margin, more misclassifications allowed.
    kernel : str, default="rbf"
        Kernel type: "linear", "poly", "rbf", or "sigmoid".
    **kwargs
        Passed to sklearn.svm.SVC.

    Example
    -------
    >>> model  = SVM(C=1.0, kernel="rbf").train(X_train, y_train)
    >>> preds  = model.predict(X_test)
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, C=1.0, kernel="rbf", **kwargs):
        # probability=True required for predict_proba / ROC-AUC
        self.model = SVC(C=C, kernel=kernel, probability=True,
                         random_state=42, **kwargs)


class NaiveBayes(_ClassificationBase):
    """Gaussian Naive Bayes classifier.

    Assumes each feature is normally distributed within each class.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Portion of the largest feature variance added to variances for
        numerical stability.

    Example
    -------
    >>> model  = NaiveBayes().train(X_train, y_train)
    >>> preds  = model.predict(X_test)
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, var_smoothing=1e-9):
        self.model = GaussianNB(var_smoothing=var_smoothing)


class KNN(_ClassificationBase):
    """K-nearest neighbours classifier.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbours to consider for majority-vote classification.
    metric : str, default="minkowski"
        Distance metric; "minkowski" with p=2 is Euclidean distance.
    **kwargs
        Passed to sklearn.neighbors.KNeighborsClassifier.

    Example
    -------
    >>> model  = KNN(n_neighbors=7).train(X_train, y_train)
    >>> preds  = model.predict(X_test)
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, n_neighbors=5, metric="minkowski", **kwargs):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors, metric=metric, **kwargs)


class XGBoost(_ClassificationBase):
    """Extreme gradient boosting classifier.

    Requires the `xgboost` package: pip install xgboost

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds (trees).
    learning_rate : float, default=0.1
        Step size shrinkage applied after each round (eta); lower → more conservative.
    max_depth : int, default=6
        Maximum depth of each tree; deeper → more complex, higher risk of overfitting.
    **kwargs
        Passed to xgboost.XGBClassifier.

    Example
    -------
    >>> model  = XGBoost(n_estimators=200, learning_rate=0.05).train(X_train, y_train)
    >>> preds  = model.predict(X_test)
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6, **kwargs):
        self.model = XGBClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, eval_metric="logloss",
            random_state=42, **kwargs)
