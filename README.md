# GROUP60_CENG313 – Hull Tactical Market Prediction

CENG313 term project. We work on the Kaggle competition
**[Hull Tactical – Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction)**.

## Team

| Name | Student ID | GitHub |
|------|------------|--------|
| Cemil Kutay Özkul | 23118080018 | [@Kutay33](https://github.com/Kutay33) |

## Competition

Every trading day we decide how much of the S&P 500 to hold. The prediction is a **position between 0 and 2**
(0 = all cash, 1 = fully invested, 2 = 2× leverage). The score is a **volatility‑adjusted Sharpe ratio**:
the Sharpe ratio is penalised when the strategy is more than 1.2× as volatile as the market or earns less
than the market. Submissions run as Kaggle notebooks through the competition's evaluation API, one day at a time.

## Dataset

- `train.csv` – one row per trading day (`date_id`), several decades of history.
  - Features grouped by prefix: **D** (dummies), **E** (macro), **I** (interest rates), **M** (market dynamics),
    **P** (price/valuation), **S** (sentiment), **V** (volatility).
  - Targets: `forward_returns`, `risk_free_rate`, `market_forward_excess_returns`.
- `test.csv` – same features plus lagged returns and an `is_scored` flag.

The data is not stored in this repository (see `.gitignore`); the notebooks download it with the Kaggle API.

## Approach

**Assignment 1 – EDA & baselines** (`assignment1_eda_baseline.ipynb`)
- EDA: missing‑value structure, target distribution, volatility clustering, feature–target correlations and their stability.
- Baselines (`src/models.py`), compared with 5-fold walk-forward validation:
  buy & hold, best past constant, volatility targeting (21/63 days), 1-year momentum, Ridge and ElasticNet on all features.

**Assignment 2 – Proposed solution** (`assignment2_solution.ipynb`)
- *(to be added)*

## Results

Mean competition score (adjusted Sharpe) over 5 walk-forward folds (validation from date_id 2348 to 9047).

| Model | CV adjusted Sharpe | Folds beating buy & hold | Kaggle public score |
|-------|-------------------:|:------------------------:|--------------------:|
| Volatility targeting (21 d) | **0.452** | 4 / 5 | |
| Momentum (252 d) | 0.436 | 3 / 5 | |
| Volatility targeting (63 d) | 0.420 | 2 / 5 | |
| Buy & hold (position = 1) | 0.418 | – | |
| ElasticNet | 0.390 | 2 / 5 | |
| Ridge | 0.327 | 1 / 5 | |
| Best past constant | 0.315 | 1 / 5 | |

## Repository layout

```
GROUP60_CENG313/
├── README.md
├── requirements.txt
├── assignment1_eda_baseline.ipynb
├── assignment2_solution.ipynb      (Assignment 2)
└── src/
    ├── data.py      loading, feature groups, missing values, time-series splits
    ├── models.py    baseline strategies (constant, volatility targeting, momentum, linear)
    └── utils.py     competition metric, backtest, walk-forward evaluation, plots
```

## How to run

1. Open the notebook in Google Colab:
   [assignment1_eda_baseline.ipynb](https://colab.research.google.com/github/Kutay33/GROUP60_CENG313/blob/main/assignment1_eda_baseline.ipynb)
2. Run the first cell. It clones the repo, asks for your `kaggle.json`
   (Kaggle → Settings → API Tokens → *Legacy API Credentials* → create a legacy API key) and downloads the data to `data/`.
   You must have joined the competition on Kaggle first.
3. Run the remaining cells.

Locally: `pip install -r requirements.txt`, put `train.csv` and `test.csv` into `data/`, then start Jupyter.

## Progress log

| Date | Work done |
|------|-----------|
| 2026-11-05 | Repository created, README skeleton, data loading and metric helpers |
| 2026-11-05 | EDA notebook |
| 2026-11-05 | Ran EDA on real data, fixed Colab setup cell, wrote EDA takeaways |
| 2026-11-09 | Baseline models and walk-forward evaluation; baseline comparison in the notebook |

## Kaggle submission history

| Date | Notebook / version | Model | Public score |
|------|--------------------|-------|-------------:|
| | | | |
