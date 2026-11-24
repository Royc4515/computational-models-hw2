# Computational Models – Exercise 2

**Due:** 29/11/2026, 23:59 · pair submission in Moodle (one of us uploads)
**Assignment:** [`docs/HW2.pdf`](docs/HW2.pdf)
**Shared report (Google Doc):** https://docs.google.com/document/d/1Xh11NT7P6wBIgs5NZknOHxbnlaIOXVwiZzIugZ5PuXo/edit

## Division of work

| Part | Owner | Notebook | Branch | Status |
|---|---|---|---|---|
| Question 1 – Linear models and maximum likelihood | Roy | `notebooks/HW2_Q1.ipynb` | `q1` | ✅ Done |
| Question 2 – Cross-validation and bias-variance | Miri | `notebooks/HW2_Q2_Q3.ipynb` | `q2-q3` | ⏳ To do |
| Question 3 – AIC and BIC | Miri | `notebooks/HW2_Q2_Q3.ipynb` | `q2-q3` | ⏳ To do |

### Question 1 (Roy) – what was done

Data: `data/HW2_q1_data.csv` – stimulus `x1` and firing rate `y1` of a neuron, N = 50.

| Task | What was done | Result |
|---|---|---|
| a | Analytical least-squares solution via the normal equations, θ = (XᵀX)⁻¹Xᵀy | θ₀ = 10.1289, θ₁ = 2.8840 |
| b | Batch gradient descent on the MSE (learning rate 0.01, 10,000 epochs, start at [0, 0]); compared with (a) | Identical to (a), difference ≈ 10⁻¹³ |
| c | MSE and R² of the fitted model | MSE = 3.3006, R² = 0.9563 |
| d | Gaussian log-likelihood with σ² = MSE, with the full calculation | LL = −100.7993 |
| e | Scatter plot of the data with the fitted line | Plot inside the notebook |

Every task has a code cell, its output, and a short explanation cell answering what the task asks.

### Questions 2 & 3 (Miri) – what is left

Data: `data/HW2_q2_data.csv` – columns `x2`, `y2`, N = 60.

The notebook `notebooks/HW2_Q2_Q3.ipynb` is ready with the setup (Drive mount, imports, data loading)
and an empty cell under each task heading:

- **Q2 a** – scatter plot of the data
- **Q2 b** – fit polynomials of degree 1, 3, 9 and plot them over the data (title, axis labels, legend)
- **Q2 c** – 5-fold cross-validation, average CV-MSE per degree
- **Q2 d** – which model under-fits / over-fits / fits best, explained with the bias-variance tradeoff
- **Q3 a** – fit the three models on the full data, MSE of each
- **Q3 b** – AIC = N ln(MSE) + 2k and BIC = N ln(MSE) + k ln(N), with k = degree + 1
- **Q3 c** – which model each criterion prefers, agreement with CV, effect of the penalty term

Same format as Question 1: code cell → output → short explanation of what the task asks.

## Repository layout

| Path | Contents |
|---|---|
| `docs/HW2.pdf` | Assignment instructions |
| `data/` | `HW2_q1_data.csv` (Q1), `HW2_q2_data.csv` (Q2 & Q3) |
| `notebooks/` | `HW2_Q1.ipynb` (Q1, done), `HW2_Q2_Q3.ipynb` (Q2 & Q3, skeleton) |

## How we work

The notebooks live in Google Drive and run in Colab; GitHub keeps a copy of each version.

**Google Drive folder:** `My Drive/Computational Models - HW2/`
```
Computational Models - HW2/
├── data/
│   ├── HW2_q1_data.csv
│   └── HW2_q2_data.csv
├── HW2_Q1.ipynb
└── HW2_Q2_Q3.ipynb
```
Each notebook reads its data from `/content/drive/MyDrive/Computational Models - HW2/data/`.
If the folder is shared with you rather than owned by you, add a shortcut to it in your My Drive
(right-click → Organize → Add shortcut) so this path works.

**Working in Colab**
1. Open the notebook from the Drive folder.
2. Runtime → Run all (approve Drive access when asked).
3. Colab saves to Drive automatically.

**Saving a version to GitHub** (manual, at milestones and before submission):
File → Save a copy in GitHub, with:

| Field | Q1 (Roy) | Q2 & Q3 (Miri) |
|---|---|---|
| Repository | `Royc4515/computational-models-hw2` | same |
| Branch | `q1` | `q2-q3` |
| File path | `notebooks/HW2_Q1.ipynb` | `notebooks/HW2_Q2_Q3.ipynb` |
| Include a link to Colab | unchecked | unchecked |

Keep the `notebooks/` prefix in the file path, otherwise Colab creates a second copy at the repository root.
When a part is finished, open a pull request from its branch into `main`.

## Submission checklist

- [ ] All notebooks run top to bottom with Run all, outputs saved
- [ ] Every task answers exactly what is asked (numbers, plots, explanations)
- [ ] PDF with explanations and calculations of all answers
- [ ] Python files (notebooks, or File → Download → Download .py)
- [ ] One of us uploads to Moodle
