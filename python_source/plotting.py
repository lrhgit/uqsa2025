import numpy as np

def plot_sobol_indices(Si, ST, ax=None):
    Nk = len(Si)
    width = 0.4
    x_tick_list = np.arange(Nk) + 1

    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax.clear()

    ax.bar(x_tick_list, Si, width, color='blue', label='Si')
    ax.bar(x_tick_list + width, ST, width, color='red', label='ST')
    ax.set_xticks(x_tick_list + width / 2)
    ax.set_xticklabels([f'x{i}' for i in x_tick_list])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Index value")
    ax.set_title("Sobol Sensitivity Indices")
    ax.legend()

    if ax is not None and hasattr(ax.figure.canvas, "draw_idle"):
        ax.figure.canvas.draw_idle()

    return ax


