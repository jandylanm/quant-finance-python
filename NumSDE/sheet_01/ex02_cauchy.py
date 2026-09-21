"""Exercise 2: Cauchy distribution via the inversion method."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qf.numsde.distributions import cauchy
from qf.numsde.plotting import plot_histogram

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    # Test run from the exercise: Cauchy(10, 0, 1)
    print(cauchy(10, 0.0, 1.0, rng))

    # Larger sample for the histogram
    sample = cauchy(10_000, 0.0, 1.0, rng)
    fig = plot_histogram(sample, mu=0.0, lam=1.0)
    fig.savefig(Path(__file__).with_name("ex02_cauchy_hist.png"), dpi=150)
    plt.show()
