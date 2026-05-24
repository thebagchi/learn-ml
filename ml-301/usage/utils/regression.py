import numpy as np
from sklearn.linear_model import LinearRegression as _LR, Ridge, Lasso
from sklearn.ensemble import (RandomForestRegressor as _RFR,
                              GradientBoostingRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class _RegressionBase:
    def train(self, X_train, y_train):
        """Fit the model on training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
        y_train : array-like of shape (n_samples,)  — continuous target values

        Returns
        -------
        self  (enables chaining: Model().train(X, y).predict(X_test))
        """
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        """Predict continuous target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Compute regression metrics on test data.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
        y_test : array-like of shape (n_samples,)

        Returns
        -------
        dict with keys:
            MAE  — mean absolute error
            MSE  — mean squared error
            RMSE — root mean squared error
            R2   — coefficient of determination (1.0 = perfect fit)
        """
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return {
            "MAE":  mean_absolute_error(y_test, y_pred),
            "MSE":  mse,
            "RMSE": np.sqrt(mse),
            "R2":   r2_score(y_test, y_pred),
        }


class LinearRegression(_RegressionBase):
    """Ordinary least squares linear regression.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to fit a bias term. Set to False if data is already centred.
    **kwargs
        Passed to sklearn.linear_model.LinearRegression.

    Example
    -------
    >>> model  = LinearRegression().train(X_train, y_train)
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, fit_intercept=True, **kwargs):
        self.model = _LR(fit_intercept=fit_intercept, **kwargs)


class RidgeRegression(_RegressionBase):
    """Ridge regression (L2 regularisation).

    Adds a penalty proportional to the sum of squared coefficients,
    which shrinks large weights and reduces overfitting.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularisation strength; larger → stronger shrinkage.
    **kwargs
        Passed to sklearn.linear_model.Ridge.

    Example
    -------
    >>> model  = RidgeRegression(alpha=0.5).train(X_train, y_train)
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, alpha=1.0, **kwargs):
        self.model = Ridge(alpha=alpha, **kwargs)


class LassoRegression(_RegressionBase):
    """Lasso regression (L1 regularisation).

    Adds a penalty proportional to the sum of absolute coefficients,
    which can drive irrelevant feature weights exactly to zero (sparse model).

    Parameters
    ----------
    alpha : float, default=1.0
        Regularisation strength; larger → more coefficients driven to zero.
    max_iter : int, default=10000
        Maximum solver iterations; increase if convergence warnings appear.
    **kwargs
        Passed to sklearn.linear_model.Lasso.

    Example
    -------
    >>> model  = LassoRegression(alpha=0.1).train(X_train, y_train)
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, alpha=1.0, max_iter=10_000, **kwargs):
        self.model = Lasso(alpha=alpha, max_iter=max_iter, **kwargs)


class RandomForestRegressor(_RegressionBase):
    """Random forest regressor (bagged ensemble of decision trees).

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees; more trees → lower variance, higher compute cost.
    max_depth : int or None, default=None
        Maximum depth of each tree; None grows until leaves are pure.
    **kwargs
        Passed to sklearn.ensemble.RandomForestRegressor.

    Example
    -------
    >>> model  = RandomForestRegressor(n_estimators=200).train(X_train, y_train)
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, n_estimators=100, max_depth=None, **kwargs):
        self.model = _RFR(n_estimators=n_estimators, max_depth=max_depth,
                          random_state=42, **kwargs)


class GradientBoosting(_RegressionBase):
    """Gradient boosting regressor (sequential ensemble of shallow trees).

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages (trees).
    learning_rate : float, default=0.1
        Contribution of each tree; lower → more trees needed but better generalisation.
    max_depth : int, default=3
        Depth of each individual tree; keep shallow (2–5) to avoid overfitting.
    **kwargs
        Passed to sklearn.ensemble.GradientBoostingRegressor.

    Example
    -------
    >>> model  = GradientBoosting(n_estimators=200, learning_rate=0.05).train(X_train, y_train)
    >>> report = model.evaluate(X_test, y_test)
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, **kwargs):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, **kwargs)
