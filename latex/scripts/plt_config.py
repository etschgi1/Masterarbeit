import matplotlib.pyplot as plt
from matplotlib import rcParams

TU_RED = "#e4154b"
TU_GREY = "#A5A5A5"
# Globale Plot-Stil-Settings
rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "axes.prop_cycle": plt.cycler(color=[
        "#e4154b", "#A5A5A5", "#007FFF", "#00D1A4", "#203746", "#d67c27",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ])
})

def use_latex():
    """
    Aktiviert LaTeX für Matplotlib-Plots.
    """
    rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}"
    })
