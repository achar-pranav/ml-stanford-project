# NBA MVP Prediction Using Machine Learning

**Course:** UE24CS352A  -  Machine Learning, Mini-Project
**Team:** _Pranav Hareesh Achar AM196_ and _Rishi J Mule AM224_  |  **Repository:** _github.com/achar-pranav/ml-standford-project_

## 1. Problem Statement

The NBA Most Valuable Player (MVP) award is decided by a panel of media voters who each submit a
ranked ballot; the player with the highest total vote share wins. The outcome is subjective, but it
is strongly associated with regular-season performance. This project builds and compares
machine-learning models that predict the MVP from per-game statistics. We frame the task as
**binary classification**  -  predict whether a player-season won the MVP  -  and convert predicted
probabilities into a **per-season ranking**, since the panel selects one winner from a small
shortlist rather than the whole league. Objectives: build a reproducible preprocessing pipeline,
compare Logistic Regression, Gaussian Naive Bayes, and Random Forest, evaluate with ranking metrics
(average/median rank, Top-1 and Top-3 accuracy), and identify the most influential statistics.

## 2. Dataset

Two public CSVs from Basketball-Reference (via the Kaggle *NBA/ABA/BAA stats* collection) are used:

| Dataset | Rows | Cols | Contents |
| --- | --- | --- | --- |
| `Player Award Shares.csv` | 3,465 | 10 | Award type, player, age, points won/max, vote share, `winner` flag |
| `Player Per Game.csv` | 33,339 | 32 | Games, minutes, shooting %, rebounds, assists, steals, blocks, points (per game) |

We filter the award file to `nba mvp` records. **Cleaning:** the statistics file lists traded
players once per team plus a combined row, so we keep only the combined row per player-season
(33,339 -> 26,932). We retain seasons 1956-2025, label MVP winners, and join on
`(season, player_id)`, giving **25,103 player-seasons** with **10 features** and a binary label.
Features: `g`, `mp_per_game`, `fg_percent`, `x3p_percent`, `ft_percent`, `trb_per_game`,
`ast_per_game`, `stl_per_game`, `blk_per_game`, `pts_per_game`. Missing values (mainly pre-1980
three-point %) are median-imputed **using only training rows**, and features are standardised for
Logistic Regression.

## 3. Approach

**Models.** Logistic Regression (interpretable linear baseline), Gaussian Naive Bayes (fast
probabilistic model), and Random Forest (300 trees, captures non-linear interactions). Because MVPs
are extremely rare (one per season), classes are highly imbalanced; we use
`class_weight="balanced"` for Logistic Regression and Random Forest and rely on ranking metrics.

**Evaluation.** (i) **Temporal holdout:** train on seasons up to 2019, test on 2020-2025 (21,749 /
3,354 player-seasons), preventing leakage. (ii) **Walk-forward back-test:** for each season
2010-2025, train on all prior seasons and predict that season. Per season we rank players by
predicted probability and report Top-1 accuracy (winner ranked first), Top-3 accuracy, and the
winner's average/median rank. To be fair, we also rank only **actual vote-getters**, mirroring the
real shortlist, and compare against a **points-per-game baseline**.

## 4. Implementation Overview

A single notebook, `NBA_MVP_Prediction.ipynb`, runs top-to-bottom in five stages: load and inspect
both datasets; clean, merge, and label; train the three models; evaluate (holdout, walk-forward,
vote-getter pool, baseline); and generate figures. The repo uses `data/` for raw CSVs, `results/`
for figures, and a `README.md` with setup/run instructions; dependencies are pinned in
`requirements.txt`.

## 5. Results

**Vote-getter pool, 2020-2025:**

| Model | Avg Rank | Median Rank | Top-1 | Top-3 |
| --- | --- | --- | --- | --- |
| Logistic Regression | 2.500 | 2.5 | 16.7% | 83.3% |
| Gaussian Naive Bayes | 2.667 | 2.0 | 16.7% | 83.3% |
| **Random Forest** | **2.333** | **1.5** | **50.0%** | **83.3%** |

Random Forest is best: it ranks the true MVP first in half the test seasons and inside the top three
in five of six. On the unrestricted holdout the points-per-game baseline reaches only 33.3% Top-1 /
33.3% Top-3, so the learned models add value beyond raw scoring. Walk-forward results broadly agree,
though some campaigns (e.g. Stephen Curry, 2015) stay hard when all league players are ranked.
Random Forest feature importance is led by points per game (0.295) and minutes per game (0.268),
then rebounds (0.153) and assists (0.135), with defensive and shooting-% features contributing
little  -  consistent with MVP winners being high-usage, high-scoring players.

## 6. Conclusions

A Random Forest over a small set of per-game statistics predicts the NBA MVP well, reaching **83.3%
Top-3** and **50% Top-1** accuracy on the realistic candidate pool and clearly beating a scoring-only
baseline. Scoring volume and playing time dominate, followed by rebounding and playmaking. Key
limitations: the small evaluation horizon (few test seasons), the subjectivity of MVP voting
(narrative, team success, and defensive reputation are not captured), and possible non-generalisation
of historical patterns. Future work could add advanced metrics (PER, win shares, team standings) and
a learning-to-rank objective that models the full ballot instead of a single binary winner.

## References

- Basketball-Reference: <https://www.basketball-reference.com/>
- Kaggle NBA/ABA/BAA dataset: <https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats>
