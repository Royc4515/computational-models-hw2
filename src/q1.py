"""Question 1 - Linear Models and Maximum Likelihood Estimation.

Model:  y = theta0 + theta1 * x + eps,   eps ~ N(0, sigma^2)

Sections:
    a) Analytical (closed-form / normal equations) OLS solution.
    b) Numerical solution with batch Gradient Descent on the MSE loss.
    c) MSE and R^2 of the fitted model.
    d) Gaussian log-likelihood, with sigma^2 = MSE.
    e) Plot of the data and the linear fit.
"""

from pathlib import Path  # OS-independent file paths; "/" joins path parts

import matplotlib  # plotting library (top-level module, to pick a backend)

matplotlib.use("Agg")  # headless backend: save figures without a display (must precede pyplot import)
import matplotlib.pyplot as plt  # the plotting API (figures, axes, scatter, plot...)
import numpy as np  # arrays, linear algebra and vectorised math
import pandas as pd  # used only to read the CSV file

ROOT_DIR = Path(__file__).resolve().parent.parent  # repo root: q1.py -> src/ -> repo/, independent of the cwd
DATA_DIR = ROOT_DIR / "data"  # folder holding the input CSV files
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"  # folder where all figures are saved

DATA_PATH = DATA_DIR / "HW2_q1_data.csv"  # full path to the Q1 dataset
FIGURE_PATH = FIGURES_DIR / "q1_linear_fit.png"  # full path of the section (e) figure

# Gradient-descent hyperparameters (section b).
LEARNING_RATE = 0.01  # step size eta; must stay below ~2/lambda_max ~= 0.029 for this data, or GD diverges
N_EPOCHS = 10_000  # number of full passes over the data (converges to ~1e-13 of the OLS solution)
INITIAL_THETA = np.array([0.0, 0.0])  # starting point [theta0, theta1]; the problem is convex, so any start works


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(path: Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load the stimulus (x1) and firing-rate (y1) columns as float arrays.

    Raises:
        ValueError: if the columns are missing or contain NaN values.
    """
    df = pd.read_csv(path)  # read the CSV into a table with columns x1, y1
    missing = {"x1", "y1"} - set(df.columns)  # set difference: required columns that are absent
    if missing:  # an empty set is False, so this triggers only if a column is missing
        raise ValueError(f"Missing columns in {path.name}: {missing}")  # fail early with a clear message
    if df[["x1", "y1"]].isna().any().any():  # first .any() per column, second .any() over the columns
        raise ValueError(f"{path.name} contains NaN values")  # NaN would silently turn every result into NaN

    x = df["x1"].to_numpy(dtype=float)  # stimulus intensities as a 1-D float array (length N)
    y = df["y1"].to_numpy(dtype=float)  # measured firing rates as a 1-D float array (length N)
    return x, y  # returned as a tuple: x, y = load_data()


def design_matrix(x: np.ndarray) -> np.ndarray:
    """Build X = [1, x] so that y_hat = X @ theta with theta = [theta0, theta1]."""
    return np.column_stack([np.ones_like(x), x])  # N x 2 matrix; the column of ones carries the intercept


def predict(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Return the linear prediction theta0 + theta1 * x."""
    return design_matrix(x) @ theta  # (N x 2) @ (2,) -> (N,): row i gives theta0 + theta1 * x_i


# --------------------------------------------------------------------------- #
# (a) Analytical solution
# --------------------------------------------------------------------------- #
def fit_analytical(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form OLS via the normal equations: theta = (X^T X)^-1 X^T y.

    np.linalg.solve is used instead of an explicit inverse for numerical
    stability. Equivalent scalar form:
        theta1 = cov(x, y) / var(x),   theta0 = mean(y) - theta1 * mean(x)
    """
    X = design_matrix(x)  # N x 2 design matrix [1, x]
    return np.linalg.solve(X.T @ X, X.T @ y)  # solve (X^T X) theta = X^T y; X^T X = [[N, Sx], [Sx, Sxx]]


# --------------------------------------------------------------------------- #
# (b) Numerical solution - batch gradient descent
# --------------------------------------------------------------------------- #
def mse_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Gradient of MSE(theta) = (1/N)||X theta - y||^2  ->  (2/N) X^T (X theta - y)."""
    residuals = X @ theta - y  # prediction errors for the current theta (length N)
    return (2.0 / len(y)) * (X.T @ residuals)  # length-2 vector: [dMSE/dtheta0, dMSE/dtheta1]


def fit_gradient_descent(
    x: np.ndarray,  # stimulus values
    y: np.ndarray,  # measured firing rates
    learning_rate: float = LEARNING_RATE,  # step size eta
    n_epochs: int = N_EPOCHS,  # number of update steps
    theta_init: np.ndarray = INITIAL_THETA,  # starting [theta0, theta1]
) -> tuple[np.ndarray, np.ndarray]:
    """Minimise the MSE with full-batch gradient descent.

    Returns:
        theta: final parameter vector [theta0, theta1].
        loss_history: MSE after each epoch (for convergence diagnostics).
    """
    if learning_rate <= 0 or n_epochs <= 0:  # input validation: both must be positive
        raise ValueError("learning_rate and n_epochs must be positive")

    X = design_matrix(x)  # build X once, outside the loop (it never changes)
    theta = theta_init.astype(float).copy()  # copy, so the in-place updates below don't modify INITIAL_THETA
    loss_history = np.empty(n_epochs)  # pre-allocated array for the MSE at each epoch

    for epoch in range(n_epochs):  # one iteration = one step using all N samples (full batch)
        theta -= learning_rate * mse_gradient(X, y, theta)  # update rule: theta <- theta - eta * grad
        loss_history[epoch] = mean_squared_error(y, X @ theta)  # record the loss after this step
        if not np.isfinite(loss_history[epoch]):  # inf/NaN means the steps overshoot and grow: divergence
            raise RuntimeError(  # stop immediately instead of returning garbage parameters
                f"Gradient descent diverged at epoch {epoch} "
                f"(learning_rate={learning_rate} is too large)"
            )

    return theta, loss_history  # final parameters and the full loss curve


# --------------------------------------------------------------------------- #
# (c) Goodness of fit
# --------------------------------------------------------------------------- #
def mean_squared_error(y: np.ndarray, y_hat: np.ndarray) -> float:
    """MSE = (1/N) * sum (y - y_hat)^2."""
    return float(np.mean((y - y_hat) ** 2))  # average squared residual; float() turns the numpy scalar into a Python float


def r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    """R^2 = 1 - SS_res / SS_tot (fraction of variance explained)."""
    ss_res = np.sum((y - y_hat) ** 2)  # variance left unexplained by the model (residual sum of squares)
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # total variance around the mean (the "always guess y-bar" baseline)
    return float(1.0 - ss_res / ss_tot)  # 1 = perfect fit, 0 = no better than the mean, <0 = worse than the mean


# --------------------------------------------------------------------------- #
# (d) Log-likelihood
# --------------------------------------------------------------------------- #
def gaussian_log_likelihood(y: np.ndarray, y_hat: np.ndarray, sigma2: float) -> float:
    """LL = sum_i log N(y_i | y_hat_i, sigma^2)
          = -N/2 * ln(2*pi*sigma^2) - SSE / (2*sigma^2).
    """
    if sigma2 <= 0:  # a variance must be positive (log of 0 / division by 0 otherwise)
        raise ValueError("sigma2 must be positive")
    n = len(y)  # number of observations N
    sse = np.sum((y - y_hat) ** 2)  # sum of squared errors: the only term that depends on theta
    return float(-0.5 * n * np.log(2 * np.pi * sigma2) - sse / (2 * sigma2))  # normalisation term + data-fit term


# --------------------------------------------------------------------------- #
# (e) Plot
# --------------------------------------------------------------------------- #
def plot_fit(x: np.ndarray, y: np.ndarray, theta: np.ndarray, path: Path = FIGURE_PATH) -> None:
    """Scatter the data and overlay the fitted regression line; save to `path`."""
    x_line = np.linspace(x.min(), x.max(), 200)  # 200 evenly spaced x values spanning the data range

    fig, ax = plt.subplots(figsize=(7, 5))  # new figure (7x5 inches) with a single set of axes
    ax.scatter(x, y, color="tab:blue", alpha=0.8, label="Data")  # the raw measurements as points
    ax.plot(  # the fitted line on top of the data
        x_line,  # x coordinates of the line
        predict(x_line, theta),  # y coordinates: the model prediction at each x
        color="tab:red",  # contrasting colour for the fit
        linewidth=2,  # thicker than the default so it stands out
        label=rf"Fit: $\hat{{y}} = {theta[0]:.3f} + {theta[1]:.3f}x$",  # legend shows the fitted equation (LaTeX)
    )
    ax.set_title("Q1 - Neuron firing rate vs. stimulus: linear fit")  # figure title
    ax.set_xlabel("Stimulus (x1)")  # x-axis label
    ax.set_ylabel("Firing rate (y1)")  # y-axis label
    ax.legend()  # draw the legend from the `label=` arguments
    ax.grid(alpha=0.3)  # faint grid to help read values
    fig.tight_layout()  # adjust margins so labels are not clipped

    path.parent.mkdir(parents=True, exist_ok=True)  # create outputs/figures/ if it doesn't exist yet
    fig.savefig(path, dpi=150)  # write the PNG file (150 dpi is sharp enough for a report)
    plt.close(fig)  # free the figure's memory


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    x, y = load_data()  # load and validate the dataset
    n = len(y)  # number of samples (50)
    print(f"Loaded N = {n} samples from {DATA_PATH.name}\n")

    # (a)
    theta_ols = fit_analytical(x, y)  # exact least-squares solution [theta0, theta1]
    print("(a) Analytical solution (normal equations)")
    print(f"    theta0 (intercept) = {theta_ols[0]:.6f}")  # :.6f prints 6 digits after the decimal point
    print(f"    theta1 (slope)     = {theta_ols[1]:.6f}\n")

    # (b)
    theta_gd, losses = fit_gradient_descent(x, y)  # numerical solution and its loss curve
    diff = np.abs(theta_gd - theta_ols)  # per-parameter distance from the analytical solution
    print("(b) Gradient descent")
    print(f"    learning rate = {LEARNING_RATE}, epochs = {N_EPOCHS}, init = {INITIAL_THETA}")  # hyperparameters
    print(f"    theta0 = {theta_gd[0]:.6f}   |diff| = {diff[0]:.2e}")  # :.2e prints scientific notation
    print(f"    theta1 = {theta_gd[1]:.6f}   |diff| = {diff[1]:.2e}")
    print(f"    final MSE = {losses[-1]:.6f}\n")  # losses[-1] is the last element: the MSE after the final epoch

    # (c)
    y_hat = predict(x, theta_ols)  # model predictions at the observed x values
    mse = mean_squared_error(y, y_hat)  # mean squared error of the fit
    r2 = r_squared(y, y_hat)  # fraction of variance explained
    print("(c) Goodness of fit")
    print(f"    MSE = {mse:.6f}")
    print(f"    R^2 = {r2:.6f}\n")

    # (d)
    ll = gaussian_log_likelihood(y, y_hat, sigma2=mse)  # LL with the noise variance set to the MSE
    ll_closed = -0.5 * n * (np.log(2 * np.pi * mse) + 1)  # same value in closed form: SSE/(2*MSE) = N/2
    print("(d) Log-likelihood with sigma^2 = MSE")
    print(f"    LL = -N/2 * ln(2*pi*MSE) - SSE/(2*MSE)")  # the general formula
    print(f"       = -{n}/2 * ln(2*pi*{mse:.4f}) - {n}/2")  # with the numbers plugged in
    print(f"       = {ll:.6f}   (closed-form check: {ll_closed:.6f})\n")  # both results must match

    # (e)
    plot_fit(x, y, theta_ols)  # draw and save the figure
    print(f"(e) Figure saved to {FIGURE_PATH.relative_to(ROOT_DIR)}")  # print the path relative to the repo root


if __name__ == "__main__":  # true only when the file is run directly, not when it is imported
    main()  # run all sections
