"""Plot helpers for the NumSDE exercises."""

import matplotlib.pyplot as plt
import numpy as np

from qf.numsde.distributions import cauchy_pdf


def plot_histogram(
    sample: np.ndarray,
    mu: float,
    lam: float,
    half_width: float = 10.0,
    bins: int = 100,
) -> plt.Figure:
    """Histogram of a Cauchy sample with the true density on top.

    Only the window [mu - half_width * lam, mu + half_width * lam] is plotted,
    as the heavy tails would otherwise stretch the x-axis.
    """
    n = sample.size
    lo, hi = mu - half_width * lam, mu + half_width * lam
    bin_width = (hi - lo) / bins

    # Histogram: weights normalize by the total n (not only by the samples
    # inside the window), so the bars are comparable to the true density
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        sample,
        bins=bins,
        range=(lo, hi),
        weights=np.full(n, 1.0 / (n * bin_width)),
        alpha=0.6,
        label=f"Sample histogram (N = {n})",
    )

    # True density for comparison
    x = np.linspace(lo, hi, 1000)
    ax.plot(x, cauchy_pdf(x, mu, lam), linewidth=2, color="red", label="True density")

    # Labels
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title(f"Cauchy distribution with mu={mu}, lambda={lam}")
    ax.legend()
    fig.tight_layout()
    return fig
