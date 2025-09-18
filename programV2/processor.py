import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic

class WindProfileProcessor:
    def __init__(self, df_or_path, method="none", **smooth_kwargs):
        if hasattr(df_or_path, 'read'):
            import pandas as pd
            self.df = pd.read_csv(df_or_path)
        else:
            self.df = df_or_path
        self.method = method
        self.smooth_kwargs = smooth_kwargs
        self.time = self.df['time'].values
        self.lat = self.df['lat'].values
        self.lon = self.df['lon'].values
        self.alt = self.df['alt'].values
        self.v_wind = None

    def process(self):
        # Гнучке згладжування — працює для ВСІХ методів (крім "none")
        if self.method != "none":
            from regularization import CoordinateRegularizer
            self.df = CoordinateRegularizer.regularize(self.df, method=self.method, **self.smooth_kwargs)
            self.lat = self.df['lat'].values
            self.lon = self.df['lon'].values
            self.alt = self.df['alt'].values
        self.compute_velocity()

    def compute_velocity(self):
        coords = list(zip(self.lat, self.lon, self.alt))
        times = self.time
        v_total = []
        if len(coords) < 2:
            self.v_wind = np.array([])
            return
        for i in range(1, len(coords)):
            p1 = (coords[i - 1][0], coords[i - 1][1])
            p2 = (coords[i][0], coords[i][1])
            h_dist = geodesic(p1, p2).meters
            v_dist = coords[i][2] - coords[i - 1][2]
            dt = times[i] - times[i - 1]
            if dt <= 0:
                v_total.append(0)
            else:
                v_total.append(((h_dist ** 2 + v_dist ** 2) ** 0.5) / dt)
        self.v_wind = np.array(v_total)

    def plot_profile(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.v_wind, self.alt[1:], label="v(z)", color="blue")
        plt.xlabel("Швидкість вітру (м/с)")
        plt.ylabel("Висота (м)")
        plt.title("Профіль швидкості вітру")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
