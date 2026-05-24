import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA as _ARIMA
from prophet import Prophet


def _mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


class ARIMA:
    """ARIMA (AutoRegressive Integrated Moving Average) time-series forecaster.

    Models a univariate time series using autoregressive (AR), differencing (I),
    and moving-average (MA) components.

    Parameters
    ----------
    order : tuple of (p, d, q), default=(1, 1, 1)
        p — number of autoregressive lags
        d — degree of differencing (1 makes a non-stationary series stationary)
        q — number of moving-average lags

    Example
    -------
    >>> model    = ARIMA(order=(1, 1, 1)).train(train_series)
    >>> forecast = model.predict(steps=len(test))
    >>> report   = model.evaluate(test.values, forecast.values)
    """
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self._fit = None

    def train(self, series):
        """Fit the ARIMA model on a time series.

        Parameters
        ----------
        series : pd.Series or array-like
            Univariate time series training data.

        Returns
        -------
        self
        """
        self._fit = _ARIMA(series, order=self.order).fit()
        return self

    def predict(self, steps):
        """Forecast future values.

        Parameters
        ----------
        steps : int
            Number of time steps to forecast beyond the training period.

        Returns
        -------
        forecast : pd.Series  — forecasted values with datetime index
        """
        return self._fit.forecast(steps=steps)

    def evaluate(self, y_true, y_pred):
        """Compute forecast accuracy metrics.

        Parameters
        ----------
        y_true : array-like  — actual test values
        y_pred : array-like  — forecasted values

        Returns
        -------
        dict with keys:
            MAE  — mean absolute error
            RMSE — root mean squared error
            MAPE — mean absolute percentage error (%)
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return {
            "MAE":  mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mse),
            "MAPE": _mape(y_true, y_pred),
        }


class ProphetModel:
    """Facebook Prophet time-series forecaster.

    Decomposes the series into trend, seasonality, and holiday components.
    Robust to missing data and shifts in trend, works well for weekly/yearly
    seasonal data.

    Parameters
    ----------
    yearly_seasonality : bool, default=False
        Whether to model yearly seasonality.
    weekly_seasonality : bool, default=True
        Whether to model weekly seasonality.
    daily_seasonality : bool, default=False
        Whether to model daily seasonality.
    **kwargs
        Passed to prophet.Prophet.

    Example
    -------
    >>> model    = ProphetModel(weekly_seasonality=True).train(train_df)
    >>> forecast = model.predict(periods=len(test_df))
    >>> report   = model.evaluate(test_df["y"].values, forecast, len(test_df))
    """
    def __init__(self, yearly_seasonality=False, weekly_seasonality=True,
                 daily_seasonality=False, **kwargs):
        self.model = Prophet(yearly_seasonality=yearly_seasonality,
                             weekly_seasonality=weekly_seasonality,
                             daily_seasonality=daily_seasonality, **kwargs)

    def train(self, train_df):
        """Fit Prophet on a DataFrame with 'ds' (date) and 'y' (value) columns.

        Parameters
        ----------
        train_df : pd.DataFrame with columns 'ds' and 'y'

        Returns
        -------
        self
        """
        self.model.fit(train_df)
        return self

    def predict(self, periods, freq="D"):
        """Generate forecast for future periods.

        Parameters
        ----------
        periods : int   — number of future periods to forecast
        freq    : str   — frequency string (e.g. "D" for daily, "W" for weekly)

        Returns
        -------
        forecast_df : pd.DataFrame  — includes 'yhat', 'yhat_lower', 'yhat_upper'
        """
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        return self.model.predict(future)

    def evaluate(self, y_true, forecast_df, n_test):
        """Compute forecast accuracy metrics from a Prophet forecast DataFrame.

        Parameters
        ----------
        y_true      : array-like  — actual test values
        forecast_df : pd.DataFrame  — returned by predict
        n_test      : int  — number of test periods (selects the last n rows of forecast_df)

        Returns
        -------
        dict with keys: MAE, RMSE, MAPE
        """
        y_pred = forecast_df["yhat"].values[-n_test:]
        mse = mean_squared_error(y_true, y_pred)
        return {
            "MAE":  mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mse),
            "MAPE": _mape(np.array(y_true), np.array(y_pred)),
        }


class _LSTMModel(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.lstm   = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :]).squeeze(-1)


class LSTMForecaster:
    """LSTM-based time-series forecaster built with PyTorch.

    Trains a single-layer LSTM that predicts the next value from a sliding
    window of past values.

    Parameters
    ----------
    hidden_size : int, default=32
        Number of LSTM hidden units.
    num_layers  : int, default=1
        Number of stacked LSTM layers.
    epochs      : int, default=20
        Training epochs.
    batch_size  : int, default=32
        Mini-batch size for gradient updates.
    lr          : float, default=1e-3
        Adam optimiser learning rate.

    Example
    -------
    >>> model  = LSTMForecaster(epochs=20).train(X_train_t, y_train_t)
    >>> y_pred = model.predict(X_test_t)
    >>> report = model.evaluate(y_test, y_pred, scaler)
    """
    def __init__(self, hidden_size=32, num_layers=1,
                 epochs=20, batch_size=32, lr=1e-3):
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.model       = None

    def train(self, X_train, y_train):
        """Train the LSTM on scaled sequence data.

        Parameters
        ----------
        X_train : torch.Tensor of shape (n_samples, seq_len, 1)  — input sequences
        y_train : torch.Tensor of shape (n_samples,)  — target next values

        Returns
        -------
        self
        """
        self.model = _LSTMModel(self.hidden_size, self.num_layers)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion  = nn.MSELoss()
        loader     = DataLoader(TensorDataset(X_train, y_train),
                                batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}  "
                      f"loss={total_loss/len(loader):.4f}")
        return self

    def predict(self, X):
        """Run inference on scaled sequence data.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, seq_len, 1)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)  — scaled predictions
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(X).numpy()

    def evaluate(self, y_true, y_pred, scaler):
        """Compute forecast metrics after inverse-scaling predictions.

        Parameters
        ----------
        y_true  : array-like  — scaled ground-truth values
        y_pred  : array-like  — scaled model predictions
        scaler  : fitted sklearn scaler  — used to invert the scaling

        Returns
        -------
        dict with keys: MAE, RMSE, MAPE  (all in original scale)
        """
        y_true_inv = scaler.inverse_transform(
            np.array(y_true).reshape(-1, 1)).flatten()
        y_pred_inv = scaler.inverse_transform(
            np.array(y_pred).reshape(-1, 1)).flatten()
        mse = mean_squared_error(y_true_inv, y_pred_inv)
        return {
            "MAE":  mean_absolute_error(y_true_inv, y_pred_inv),
            "RMSE": np.sqrt(mse),
            "MAPE": _mape(y_true_inv, y_pred_inv),
        }
