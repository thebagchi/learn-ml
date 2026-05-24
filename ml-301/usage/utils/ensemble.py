from sklearn.ensemble import (BaggingClassifier, AdaBoostClassifier,
                              StackingClassifier, RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from .classification import _ClassificationBase


class Bagging(_ClassificationBase):
    """Bagging (Bootstrap Aggregating) classifier.

    Trains multiple decision trees on random bootstrap samples of the
    training data and combines their predictions by majority vote.

    Parameters
    ----------
    n_estimators : int, default=50
        Number of base estimators (trees) in the ensemble.
    max_samples : float or int, default=1.0
        Fraction (or count) of training samples drawn per base estimator.
        Values < 1.0 increase diversity between trees.
    max_features : float or int, default=1.0
        Fraction (or count) of features considered per base estimator.
        Values < 1.0 further decorrelate the trees.
    **kwargs
        Passed to sklearn.ensemble.BaggingClassifier.

    Example
    -------
    >>> model  = Bagging(n_estimators=100).train(X_train, y_train)
    >>> preds  = model.predict(X_test)
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, n_estimators=50, max_samples=1.0, max_features=1.0, **kwargs):
        self.model = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators, max_samples=max_samples,
            max_features=max_features, random_state=42, **kwargs)


class AdaBoost(_ClassificationBase):
    """AdaBoost (Adaptive Boosting) classifier.

    Sequentially trains weak learners, giving higher weight to samples
    misclassified in previous rounds. Final prediction is a weighted vote.

    Parameters
    ----------
    n_estimators : int, default=100
        Maximum number of estimators at which boosting is terminated.
    learning_rate : float, default=1.0
        Weight applied to each estimator's contribution; trading off with
        n_estimators — lower rate needs more estimators.
    **kwargs
        Passed to sklearn.ensemble.AdaBoostClassifier.

    Example
    -------
    >>> model  = AdaBoost(n_estimators=200, learning_rate=0.5).train(X_train, y_train)
    >>> preds  = model.predict(X_test)
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, n_estimators=100, learning_rate=1.0, **kwargs):
        self.model = AdaBoostClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            random_state=42, **kwargs)


class Stacking(_ClassificationBase):
    """Stacking ensemble classifier.

    Trains a fixed set of base learners (Random Forest, SVM, KNN) and
    then fits a Logistic Regression meta-learner on their cross-validated
    out-of-fold predictions.

    Parameters
    ----------
    cv : int, default=5
        Number of cross-validation folds used to generate the meta-features
        for the final estimator. More folds → less data leakage, higher cost.
    **kwargs
        Passed to sklearn.ensemble.StackingClassifier.

    Example
    -------
    >>> model  = Stacking(cv=5).train(X_train, y_train)
    >>> preds  = model.predict(X_test)
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, cv=5, **kwargs):
        estimators = [
            ("rf",  RandomForestClassifier(n_estimators=100, random_state=42)),
            ("svm", SVC(probability=True, random_state=42)),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ]
        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=cv, **kwargs)
