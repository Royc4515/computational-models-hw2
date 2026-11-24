# Computational Models – Exercise 2

**Due:** 29/11/2026, 23:59 · pair submission in Moodle
**Assignment:** [`docs/HW2.pdf`](docs/HW2.pdf)
**Shared report (Google Doc):** https://docs.google.com/document/d/1Xh11NT7P6wBIgs5NZknOHxbnlaIOXVwiZzIugZ5PuXo/edit

## Division of work

| Part | Owner |
|---|---|
| Question 1 – Linear models and maximum likelihood | Roy |
| Questions 2 & 3 – Cross-validation, bias-variance, AIC / BIC | Miri |

## Question 1 (Roy)

Notebook: `notebooks/HW2_Q1.ipynb` (also in Google Drive under `Computational Models - HW2/`).
Data: `data/HW2_q1_data.csv` – stimulus `x1` and firing rate `y1` of a neuron, N = 50.

| Task | What was done | Result |
|---|---|---|
| a | Analytical least-squares solution via the normal equations, θ = (XᵀX)⁻¹Xᵀy | θ₀ = 10.1289, θ₁ = 2.8840 |
| b | Batch gradient descent on the MSE (learning rate 0.01, 10,000 epochs, start at [0, 0]); compared with (a) | Identical to (a), difference ≈ 10⁻¹³ |
| c | MSE and R² of the fitted model | MSE = 3.3006, R² = 0.9563 |
| d | Gaussian log-likelihood with σ² = MSE, with the full calculation | LL = −100.7993 |
| e | Scatter plot of the data with the fitted line | Plot inside the notebook |

Each task has a code cell, its output, and a short explanation of what the task asks.

## Questions 2 & 3 (Miri)

Data: `data/HW2_q2_data.csv`.

## Repository layout

| Path | Contents |
|---|---|
| `docs/HW2.pdf` | Assignment instructions |
| `data/` | `HW2_q1_data.csv` (Q1), `HW2_q2_data.csv` (Q2 & Q3) |
| `notebooks/` | `HW2_Q1.ipynb` (Q1) |
