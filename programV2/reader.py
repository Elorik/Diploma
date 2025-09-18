import pandas as pd
import numpy as np
import os

def auto_map_columns(df):
    column_mapping = {
        'time': ['time', 'час', 'Час (UTC)', 'datetime'],
        'lat': ['lat', 'широта', 'Широта (град.)'],
        'lon': ['lon', 'довгота', 'Довгота (град.)'],
        'alt': ['alt', 'висота', 'Висота (м)', 'HGHT']
    }
    for standard, candidates in column_mapping.items():
        for name in candidates:
            if name in df.columns:
                df.rename(columns={name: standard}, inplace=True)
                break
    return df

class WindDataReader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".csv")
        ]

    def load_all(self):
        datasets = []
        for file in self.files:
            try:
                # 1) Пропускаємо рядки-коментарі
                df = pd.read_csv(file, comment='#')
                # 2) Нормалізуємо назви стовпців
                df = auto_map_columns(df)
                # 3) Відкидаємо будь-які рядки, де нема time/lat/lon/alt
                df = df.dropna(subset=['lat', 'lon', 'alt'])
                # 4) Видаляємо повністю дубльовані спостереження
                df = df.drop_duplicates(subset=['time', 'lat', 'lon', 'alt'])
                
                # Перетворення time у секунди
                if 'time' in df.columns:
                    try:
                        df['time'] = pd.to_datetime(df['time'])
                        df['time'] = (
                            df['time'] - df['time'].iloc[0]
                        ).dt.total_seconds()
                    except:
                        df['time'] = np.arange(0, len(df) * 5, 5)
                else:
                    df['time'] = np.arange(0, len(df) * 5, 5)

                # Додаємо лише якщо всі ключові колонки є
                if all(col in df.columns for col in ['time', 'lat', 'lon', 'alt']):
                    datasets.append((os.path.basename(file), df))
            except Exception as e:
                print(f"⚠️ Помилка при обробці {file}: {e}")
        return datasets
