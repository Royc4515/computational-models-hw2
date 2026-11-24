"""Question 1 - Linear Models and Maximum Likelihood Estimation.

Model:  y = theta0 + theta1 * x + eps,   eps ~ N(0, sigma^2)

Sections:
    a) Analytical (closed-form / normal equations) OLS solution.
    b) Numerical solution with batch Gradient Descent on the MSE loss.
    c) MSE and R^2 of the fitted model.
    d) Gaussian log-likelihood, with sigma^2 = MSE.
    e) Plot of the data and the linear fit.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: save figures without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"

DATA_PATH = DATA_DIR / "HW2_q1_data.csv"
FIGURE_PATH = FIGURES_DIR / "q1_linear_fit.png"

# Gradient-descent hyperparameters (section b).
LEARNING_RATE = 0.01
N_EPOCHS = 10_000
INITIAL_THETA = np.array([0.0, 0.0])


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(path: Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load the stimulus (x1) and firing-rate (y1) columns as float arrays.

    Raises:
        ValueError: if the columns are missing or contain NaN values.
    """
    df = pd.read_csv(path)
    missing = {"x1", "y1"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")
    if df[["x1", "y1"]].isna().any().any():
        raise ValueError(f"{path.name} contains NaN values")

    x = df["x1"].to_numpy(dtype=float)
    y = df["y1"].to_numpy(dtype=float)
    return x, y


def design_matrix(x: np.ndarray) -> np.ndarray:
    """Build X = [1, x] so that y_hat = X @ theta with theta = [theta0, theta1]."""
    return np.column_stack([np.ones_like(x), x])


def predict(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Return the linear prediction theta0 + theta1 * x."""
    return design_matrix(x) @ theta


# --------------------------------------------------------------------------- #
# (a) Analytical solution
# --------------------------------------------------------------------------- #
def fit_analytical(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form OLS via the normal equations: theta = (X^T X)^-1 X^T y.

    np.linalg.solve is used instead of an explicit inverse for numerical
    stability. Equivalent scalar form:
        theta1 = cov(x, y) / var(x),   theta0 = mean(y) - theta1 * mean(x)
    """
    X = design_matrix(x)
    return np.linalg.solve(X.T @ X, X.T @ y)


# --------------------------------------------------------------------------- #
# (b) Numerical solution - batch gradient descent
# --------------------------------------------------------------------------- #
def mse_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Gradient of MSE(theta) = (1/N)||X theta - y||^2  ->  (2/N) X^T (X theta - y)."""
    residuals = X @ theta - y
    return (2.0 / len(y)) * (X.T @ residuals)


def fit_gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = LEARNING_RATE,
    n_epochs: int = N_EPOCHS,
    theta_init: np.ndarray = INITIAL_THETA,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimise the MSE with full-batch gradient descent.

    Returns:
        theta: final parameter vector [theta0, theta1].
        loss_history: MSE after each epoch (for convergence diagnostics).
    """
    if learning_rate <= 0 or n_epochs <= 0:
        raise ValueError("learning_rate and n_epochs must be positive")

    X = design_matrix(x)
    theta = theta_init.astype(float).copy()
    loss_history = np.empty(n_epochs)

    for epoch in range(n_epochs):
        theta -= learning_rate * mse_gradient(X, y, theta)
        loss_history[epoch] = mean_squared_error(y, X @ theta)
        if not np.isfinite(loss_history[epoch]):
            raise RuntimeError(
                f"Gradient descent diverged at epoch {epoch} "
                f"(learning_rate={learning_rate} is too large)"
            )

    return theta, loss_history


# --------------------------------------------------------------------------- #
# (c) Goodness of fit
# --------------------------------------------------------------------------- #
def mean_squared_error(y: np.ndarray, y_hat: np.ndarray) -> float:
    """MSE = (1/N) * sum (y - y_hat)^2."""
    return float(np.mean((y - y_hat) ** 2))


def r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    """R^2 = 1 - SS_res / SS_tot (fraction of variance explained)."""
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot)


# --------------------------------------------------------------------------- #
# (d) Log-likelihood
# --------------------------------------------------------------------------- #
def gaussian_log_likelihood(y: np.ndarray, y_hat: np.ndarray, sigma2: float) -> float:
    """LL = sum_i log N(y_i | y_hat_i, sigma^2)
          = -N/2 * ln(2*pi*sigma^2) - SSE / (2*sigma^2).
    """
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")
    n = len(y)
    sse = np.sum((y - y_hat) ** 2)
    return float(-0.5 * n * np.log(2 * np.pi * sigma2) - sse / (2 * sigma2))


# --------------------------------------------------------------------------- #
# (e) Plot
# --------------------------------------------------------------------------- #
def plot_fit(x: np.ndarray, y: np.ndarray, theta: np.ndarray, path: Path = FIGURE_PATH) -> None:
    """Scatter the data and overlay the fitted regression line; save to `path`."""
    x_line = np.linspace(x.min(), x.max(), 200)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, color="tab:blue", alpha=0.8, label="Data")
    ax.plot(
        x_line,
        predict(x_line, theta),
        color="tab:red",
        linewidth=2,
        label=rf"Fit: $\hat{{y}} = {theta[0]:.3f} + {theta[1]:.3f}x$",
    )
    ax.set_title("Q1 - Neuron firing rate vs. stimulus: linear fit")
    ax.set_xlabel("Stimulus (x1)")
    ax.set_ylabel("Firing rate (y1)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    x, y = load_data()
    n = len(y)
    print(f"Loaded N = {n} samples from {DATA_PATH.name}\n")

    # (a)
    theta_ols = fit_analytical(x, y)
    print("(a) Analytical solution (normal equations)")
    print(f"    theta0 (intercept) = {theta_ols[0]:.6f}")
    print(f"    theta1 (slope)     = {theta_ols[1]:.6f}\n")

    # (b)
    theta_gd, losses = fit_gradient_descent(x, y)
    diff = np.abs(theta_gd - theta_ols)
    print("(b) Gradient descent")
    print(f"    learning rate = {LEARNING_RATE}, epochs = {N_EPOCHS}, init = {INITIAL_THETA}")
    print(f"    theta0 = {theta_gd[0]:.6f}   |diff| = {diff[0]:.2e}")
    print(f"    theta1 = {theta_gd[1]:.6f}   |diff| = {diff[1]:.2e}")
    print(f"    final MSE = {losses[-1]:.6f}\n")

    # (c)
    y_hat = predict(x, theta_ols)
    mse = mean_squared_error(y, y_hat)
    r2 = r_squared(y, y_hat)
    print("(c) Goodness of fit")
    print(f"    MSE = {mse:.6f}")
    print(f"    R^2 = {r2:.6f}\n")

    # (d)
    ll = gaussian_log_likelihood(y, y_hat, sigma2=mse)
    ll_closed = -0.5 * n * (np.log(2 * np.pi * mse) + 1)  # SSE/(2*MSE) = N/2
    print("(d) Log-likelihood with sigma^2 = MSE")
    print(f"    LL = -N/2 * ln(2*pi*MSE) - SSE/(2*MSE)")
    print(f"       = -{n}/2 * ln(2*pi*{mse:.4f}) - {n}/2")
    print(f"       = {ll:.6f}   (closed-form check: {ll_closed:.6f})\n")

    # (e)
    plot_fit(x, y, theta_ols)
    print(f"(e) Figure saved to {FIGURE_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
