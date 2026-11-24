# Computational Models – Exercise 2

**Shared report (Google Doc):** https://docs.google.com/document/d/1Xh11NT7P6wBIgs5NZknOHxbnlaIOXVwiZzIugZ5PuXo/edit

## Directory map

| Path | Purpose |
|---|---|
| `docs/` | Assignment instructions (`HW2.pdf`) |
| `data/` | Input datasets: `HW2_q1_data.csv` (Q1), `HW2_q2_data.csv` (Q2 & Q3) |
| `src/` | Source code, one file per part |
| `notebooks/` | Colab notebooks, mirrored in Google Drive under `Computational Models - HW2/`: `HW2_Q1.ipynb` (Question 1: code, outputs and explanations) |
| `outputs/figures/` | All generated plots (save every figure here) |

## Division of work

| Part | File | Topics |
|---|---|---|
| Part 1 | `notebooks/HW2_Q1.ipynb` | Linear models, OLS, gradient descent, MSE / R², log-likelihood |
| Parts 2 & 3 | `src/q2_q3.py` | Polynomial fits, 5-fold CV, bias-variance, AIC / BIC |

Each part writes its numbers and figures into the matching section of the shared Google Doc.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run a part from anywhere (paths resolve relative to the repo):

```bash
python src/q2_q3.py
```

Save generated plots to `outputs/figures/` using the names referenced in the report
(`q2_scatter.png`, `q2_polynomial_fits.png`).
