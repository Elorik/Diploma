import matplotlib.pyplot as plt

def plot_profile(processor_raw, processor_smooth):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Перевірка, що в обох є дані (і не порожні)
    if processor_raw.v_wind is not None and len(processor_raw.v_wind) > 0:
        ax.plot(processor_raw.v_wind, processor_raw.alt[1:], label="Без згладжування", linestyle="--", color="gray")
    if processor_smooth.v_wind is not None and len(processor_smooth.v_wind) > 0:
        ax.plot(processor_smooth.v_wind, processor_smooth.alt[1:], label="Після згладжування", color="blue")

    ax.set_xlabel("Швидкість вітру (м/с)")
    ax.set_ylabel("Висота (м)")
    ax.set_title("Профіль швидкості вітру")
    ax.grid(True)
    ax.legend()
    return fig
