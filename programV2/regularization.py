import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.linalg import solve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error

class CoordinateRegularizer:
    @staticmethod
    def apply_savgol(arr, window_length=11, polyorder=2, **kwargs):
        # Додаємо захист для малих масивів і для непарного window_length
        window_length = max(3, int(window_length))
        if window_length % 2 == 0:
            window_length += 1  # Savgol потребує непарне
        if len(arr) < window_length:
            return arr
        return savgol_filter(arr, window_length=window_length, polyorder=polyorder)

    @staticmethod
    def apply_spline(x, y, smooth_factor=1, **kwargs):
        if len(x) < 4:
            return y
        spline = UnivariateSpline(x, y, s=smooth_factor)
        return spline(x)

    @staticmethod
    def apply_moving_average(arr, window=5, **kwargs):
        window = max(2, int(window))
        return pd.Series(arr).rolling(window=window, center=True, min_periods=1).mean().values

    @staticmethod
    def apply_kalman_filter(data, process_variance=1e-3, measurement_variance=1.0, **kwargs):
        n = len(data)
        x_hat = np.zeros(n)
        P = np.zeros(n)
        x_hat[0] = data[0]
        P[0] = 1.0
        for k in range(1, n):
            x_pred = x_hat[k-1]
            P_pred = P[k-1] + process_variance
            K = P_pred / (P_pred + measurement_variance)
            x_hat[k] = x_pred + K * (data[k] - x_pred)
            P[k] = (1 - K) * P_pred
        return x_hat

    @staticmethod
    def apply_tikhonov_regularization(y, alpha=1e-2, **kwargs):
        n = len(y)
        D = np.eye(n) - np.eye(n, k=1)
        D = D[:-1]
        A = np.eye(n) + alpha * D.T @ D
        return solve(A, y)

    @staticmethod
    def regularize(df, method="savgol", **kwargs):
        df_smooth = df.copy()
        t = df['time'].values
        for col in ['lat', 'lon', 'alt']:
            if method == "savgol":
                df_smooth[col] = CoordinateRegularizer.apply_savgol(df[col].values, 
                                                                   window_length=kwargs.get("window_length", 11), 
                                                                   polyorder=kwargs.get("polyorder", 2))
            elif method == "spline":
                df_smooth[col] = CoordinateRegularizer.apply_spline(t, df[col].values, 
                                                                   smooth_factor=kwargs.get("smooth_factor", 1.0))
            elif method == "ma":
                df_smooth[col] = CoordinateRegularizer.apply_moving_average(df[col].values, 
                                                                           window=kwargs.get("window", 5))
            elif method == "kalman":
                df_smooth[col] = CoordinateRegularizer.apply_kalman_filter(df[col].values, 
                                                                           process_variance=kwargs.get("process_variance", 1e-3), 
                                                                           measurement_variance=kwargs.get("measurement_variance", 1.0))
            elif method == "tikhonov":
                df_smooth[col] = CoordinateRegularizer.apply_tikhonov_regularization(df[col].values, 
                                                                                     alpha=kwargs.get("alpha", 1e-2))
            elif method == "none":
                pass # нічого не робимо
        return df_smooth

    @staticmethod
    def available_methods():
        return ["none", "savgol", "spline", "ma", "kalman", "tikhonov"]

def compute_metrics(y_true, y_pred):
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MaxE": max_error(y_true, y_pred)
    }
