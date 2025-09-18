import numpy as np
import streamlit as st
import pandas as pd
import os
from reader import WindDataReader
from processor import WindProfileProcessor
from visualization import plot_profile
from regularization import CoordinateRegularizer, compute_metrics

st.set_page_config(page_title="Wind Profile Analysis")
st.title("🌬️ Аналіз швидкості вітру за координатами")

folder = st.text_input("Шлях до папки з CSV:", value="real_data")

if os.path.isdir(folder):
    reader = WindDataReader(folder)
    datasets = reader.load_all()

    if datasets:
        file_names = [name for name, _ in datasets]
        selected = st.selectbox("Оберіть файл:", file_names)
        df = dict(datasets)[selected]
        if 'Істинна швидкість (м/с)' in df.columns:
            df.rename(columns={'Істинна швидкість (м/с)': 'v_true'}, inplace=True)

        available_methods = CoordinateRegularizer.available_methods()
        method = st.selectbox("Метод згладжування:", available_methods, index=1)

        # --- Параметри згладжування ---
        smooth_kwargs = {}
        if method == "savgol":
            window_length = st.number_input("Довжина вікна (Savgol)", min_value=3, max_value=101, value=11, step=2)
            polyorder = st.number_input("Степінь полінома (Savgol)", min_value=1, max_value=5, value=2)
            smooth_kwargs = {"window_length": window_length, "polyorder": polyorder}
        elif method == "spline":
            smooth_factor = st.slider("Коефіцієнт згладжування (Spline)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            smooth_kwargs = {"smooth_factor": smooth_factor}
        elif method == "ma":
            window = st.number_input("Довжина вікна (MA)", min_value=2, max_value=101, value=5, step=1)
            smooth_kwargs = {"window": window}
        elif method == "kalman":
            process_variance = st.number_input("Process variance (Kalman)", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6f")
            measurement_variance = st.number_input("Measurement variance (Kalman)", min_value=1e-3, max_value=10.0, value=1.0)
            smooth_kwargs = {"process_variance": process_variance, "measurement_variance": measurement_variance}
        elif method == "tikhonov":
            alpha = st.number_input("Коефіцієнт регуляризації (Tikhonov)", min_value=1e-5, max_value=1.0, value=1e-2, format="%.5f")
            smooth_kwargs = {"alpha": alpha}
        # --- Кінець параметрів ---

        # Обробка профілю без згладжування (для порівняння)
        processor_raw = WindProfileProcessor(df, method="none")
        processor_raw.process()

        # Обробка профілю з обраним методом і параметрами
        processor = WindProfileProcessor(df, method=method, **smooth_kwargs)
        processor.process()

        # Графік "до/після"
        st.pyplot(plot_profile(processor_raw, processor))

        st.dataframe(pd.DataFrame({
            "Висота (м)": processor.alt[1:],
            "Швидкість вітру (м/с)": processor.v_wind
        }))

        # === МЕТРИКИ: для синтетики — класичні, для реальних — статистика ===
        if "v_true" in df.columns:
            metrics = compute_metrics(df["v_true"].values, processor.v_wind)
            st.subheader("📊 Метрики точності для обраного методу")
            st.table(pd.DataFrame([metrics]))

            st.subheader("📋 Порівняння всіх методів")
            all_metrics = []
            for m in available_methods:
                proc = WindProfileProcessor(df.copy(), method=m)
                proc.process()
                method_metrics = compute_metrics(df["v_true"].values, proc.v_wind)
                method_metrics = {"Метод": m, **method_metrics}
                all_metrics.append(method_metrics)

            df_metrics = pd.DataFrame(all_metrics).set_index("Метод")
            st.dataframe(df_metrics)
        else:
            # Для реальних профілів — статистики та порівняння до/після згладжування
            st.subheader("📊 Статистичні метрики швидкості (згладженої):")
            if len(processor.v_wind) > 0:
                st.markdown(f"**Середнє**: `{np.mean(processor.v_wind):.3f}` м/с")
                st.markdown(f"**Медіана**: `{np.median(processor.v_wind):.3f}` м/с")
                st.markdown(f"**Std**: `{np.std(processor.v_wind):.3f}` м/с")
                st.markdown(f"**Максимум**: `{np.max(processor.v_wind):.3f}` м/с")
                st.markdown(f"**Мінімум**: `{np.min(processor.v_wind):.3f}` м/с")
            else:
                st.info("Недостатньо даних для розрахунку статистик.")

            st.subheader("📊 Статистичні метрики швидкості (без згладжування):")
            if len(processor_raw.v_wind) > 0:
                st.markdown(f"**Середнє**: `{np.mean(processor_raw.v_wind):.3f}` м/с")
                st.markdown(f"**Медіана**: `{np.median(processor_raw.v_wind):.3f}` м/с")
                st.markdown(f"**Std**: `{np.std(processor_raw.v_wind):.3f}` м/с")
                st.markdown(f"**Максимум**: `{np.max(processor_raw.v_wind):.3f}` м/с")
                st.markdown(f"**Мінімум**: `{np.min(processor_raw.v_wind):.3f}` м/с")
            else:
                st.info("Недостатньо даних для розрахунку статистик.")

            st.subheader("📊 Відносне порівняння (до/після згладжування):")
            if len(processor_raw.v_wind) > 0 and len(processor.v_wind) > 0:
                min_len = min(len(processor_raw.v_wind), len(processor.v_wind))
                abs_diff = np.abs(processor_raw.v_wind[:min_len] - processor.v_wind[:min_len])
                st.markdown(f"**Середнє абсолютне відхилення:** `{np.mean(abs_diff):.3f}` м/с")
                st.markdown(f"**Максимальне абсолютне відхилення:** `{np.max(abs_diff):.3f}` м/с")
            else:
                st.info("Недостатньо даних для порівняння профілів.")

        # Вивід попередження, якщо замало точок
        if len(processor.v_wind) == 0:
            st.warning("У файлі замало точок для розрахунку швидкості вітру!")

    else:
        st.warning("У папці не знайдено придатних CSV-файлів.")
else:
    st.warning("Папка не знайдена або недоступна.")
