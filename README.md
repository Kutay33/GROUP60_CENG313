# GROUP60_CENG313 – Hull Tactical Market Prediction

CENG313 (Introduction to Data Science) course project, **Group 60**. I work on the Kaggle competition
**[Hull Tactical – Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction)**.

> **Competition closed — evaluated on a held-out split.**
> Kaggle has disabled late submissions for this competition, so the score below is the competition metric on the
> last 20 % of `train.csv` (see *Current score*).

## Team

| Name | Student ID | GitHub |
|------|------------|--------|
| Cemil Kutay Özkul | 23118080018 | [@Kutay33](https://github.com/Kutay33) |

## Current score

| Evaluation | Model | Adjusted Sharpe (competition metric) |
|------------|-------|-------------------------------------:|
| Held-out split (last 20 % of `train.csv`, date_id 7238–9047) | Volatility targeting (21 d) | **0.691** |
| Held-out split, benchmark | Buy & hold (position = 1) | 0.624 |
| 5-fold walk-forward validation (date_id 2348–9047) | Volatility targeting (21 d) | 0.452 |

**Comparison with the leaderboard:** the final leaderboard is led by scores of 3.85 and 3.84; the top 8 teams score
3.23–3.85 (screenshot `submissions/leaderboard_top.png`). They were measured on a short live period after the
competition ended, while our 0.691 covers ≈7 years of held-out data, so the numbers are not directly comparable.

## Competition

Every trading day the model decides how much of the S&P 500 to hold. The prediction is a **position between 0 and 2**
(0 = all cash, 1 = fully invested, 2 = 2× leverage). The score is a **volatility‑adjusted Sharpe ratio**:
the Sharpe ratio is penalised when the strategy is more than 1.2× as volatile as the market or earns less
than the market. Submissions run as Kaggle notebooks through the competition's evaluation API, one day at a time.

## Dataset

- `train.csv` – one row per trading day (`date_id` 0–9047), 94 anonymised features.
  - Features grouped by prefix: **D** (dummies), **E** (macro), **I** (interest rates), **M** (market dynamics),
    **P** (price/valuation), **S** (sentiment), **V** (volatility).
  - Targets: `forward_returns`, `risk_free_rate`, `market_forward_excess_returns`.
- `test.csv` – same features plus lagged returns and an `is_scored` flag.

### How to download the data

The data is **not** stored in this repository (`data/` is in `.gitignore`).

1. Join the competition on Kaggle and accept the rules (*Data* tab → *Join Competition*).
2. Either download it manually (*Data* tab → *Download All*) and unzip `train.csv` and `test.csv` into `data/`,
3. or use the Kaggle API (needs `kaggle.json`: Kaggle → Settings → API Tokens → *Legacy API Credentials*):
   ```bash
   kaggle competitions download -c hull-tactical-market-prediction -p data
   unzip data/hull-tactical-market-prediction.zip -d data
   ```
   On Colab the first cell of the notebook does this automatically.

## Approach

**Assignment 1 – EDA & baseline** (`assignment1_eda_baseline.ipynb`)
1. Problem definition – task, target and the volatility-adjusted Sharpe metric.
2. Data overview – shape, dtypes, `describe()`, missing values (time-dependent), duplicates.
3. EDA – target distribution and balance, outliers, market growth and volatility regimes, volatility clustering,
   feature distributions, feature–target correlations and their stability, multicollinearity.
4. Preprocessing – drop mostly-empty early rows, lagged returns, forward-fill + median imputation, scaling for linear models.
5. Baselines (`src/models.py`) with 5-fold walk-forward validation: buy & hold, best past constant,
   volatility targeting (21/63 days), 1-year momentum, Ridge, ElasticNet, LightGBM.
6. Held-out evaluation on the last 20 % (late submissions disabled) and a prepared Kaggle submission notebook.
7. Conclusion – scaling exposure by recent volatility beats buy & hold; predicting the return direction from the raw
   features does not.

**Assignment 2 – Proposed solution** (`assignment2_solution.ipynb`)
- *(to be added)*

## Results

Competition metric (adjusted Sharpe). Walk-forward: mean over 5 folds (validation from date_id 2348 to 9047).
Hold-out: trained on the first 80 % of `train.csv`, scored on the last 20 %.

| Model | Walk-forward CV | Folds beating buy & hold | Hold-out (last 20 %) |
|-------|----------------:|:------------------------:|---------------------:|
| Volatility targeting (21 d) | **0.452** | 4 / 5 | **0.691** |
| Momentum (252 d) | 0.436 | 3 / 5 | 0.575 |
| Volatility targeting (63 d) | 0.420 | 2 / 5 | 0.547 |
| Buy & hold (position = 1) | 0.418 | – | 0.624 |
| ElasticNet | 0.390 | 2 / 5 | 0.524 |
| LightGBM | 0.385 | 1 / 5 | 0.521 |
| Ridge | 0.327 | 1 / 5 | 0.398 |
| Best past constant | 0.315 | 1 / 5 | 0.618 |

## Repository layout

```
GROUP60_CENG313/
├── README.md
├── requirements.txt
├── .gitignore                       excludes data/
├── assignment1_eda_baseline.ipynb   Assignment 1: EDA & baseline
├── assignment2_solution.ipynb       Assignment 2 (to be added)
├── kaggle_submission.ipynb          Kaggle notebook for the evaluation API
├── src/
│   ├── __init__.py
│   ├── data.py      loading, feature groups, missing values, lagged returns, hold-out and walk-forward splits
│   ├── models.py    baseline strategies (constant, volatility targeting, momentum, linear, LightGBM), online predictor
│   └── utils.py     competition metric, backtest, walk-forward and hold-out evaluation, plots
└── submissions/     late_submission_disabled.png, leaderboard_top.png
```

## How to run

1. Open the notebook in Google Colab:
   [assignment1_eda_baseline.ipynb](https://colab.research.google.com/github/ceng313-2026/GROUP60_CENG313/blob/main/assignment1_eda_baseline.ipynb)
   (it is also readable directly on GitHub – all outputs are saved).
2. Run the first cell. It clones the repo, asks for your `kaggle.json` and downloads the data to `data/`.
3. Run the remaining cells (*Runtime → Run all*).

Locally: `pip install -r requirements.txt`, put `train.csv` and `test.csv` into `data/`, then start Jupyter.

**Kaggle submission:** see the instructions at the top of `kaggle_submission.ipynb`
(upload `src/` as a Kaggle Dataset, import the notebook into the competition, turn internet off, submit).

## Progress log

| Date | Work done |
|------|-----------|
| 2026-11-05 | Repository created, README skeleton, data loading and metric helpers |
| 2026-11-05 | EDA notebook |
| 2026-11-05 | Ran EDA on real data, fixed Colab setup cell, wrote EDA takeaways |
| 2026-11-09 | Baseline models and walk-forward evaluation; baseline comparison in the notebook |
| 2026-11-10 | LightGBM baseline, online predictor for the evaluation API, Kaggle submission notebook |
| 2026-11-12 | Moved the repository to the course organization; notebook restructured into the 7 required sections; hold-out evaluation; `submissions/` |

## Kaggle submission history

| Date | Notebook / version | Model | Score |
|------|--------------------|-------|------:|
| 2026-11-10 | [kaggle_submission.ipynb](https://www.kaggle.com/code/cemilkutayzkul/notebookf783772543) – Version 2 | Volatility targeting (21 d) | n/a – submissions disabled |
| 2026-11-12 | Held-out evaluation (last 20 % of `train.csv`) | Volatility targeting (21 d) | 0.691 |

Late submissions are disabled for this competition ("Submissions have been disabled for this competition", checked on
2026-11-10; screenshot in `submissions/`). The submission notebook was verified with the official `kaggle_evaluation`
gateway in a local dry run, and the model is evaluated on a held-out split instead.
