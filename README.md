# NBA MVP Prediction Using Machine Learning

## Project Overview

This project explores the use of machine learning to predict the NBA Most Valuable Player (MVP) using regular-season player statistics. It compares three classification algorithms and evaluates their ability to rank MVP candidates.

## Objectives

- Analyze historical NBA player statistics and MVP voting records.
- Prepare and preprocess the datasets for machine learning.
- Train and compare Logistic Regression, Naive Bayes, and Random Forest models.
- Evaluate model performance using ranking-based metrics.
- Identify the player statistics that contribute most to the Random Forest model's predictions.

## Dataset

The project uses two CSV files:

- `Player Award Shares.csv` — contains historical NBA award voting information.
- `Player Per Game.csv` — contains player statistics for individual seasons.

The datasets cover NBA seasons from 1956 to 2025. Data is joined using player identifiers and seasons.

## Machine Learning Models

1. **Logistic Regression** — a linear classification baseline.
2. **Gaussian Naive Bayes** — a probabilistic classification model.
3. **Random Forest** — an ensemble of decision trees.

## Evaluation

The models are compared using average rank, median rank, Top-1 accuracy, and Top-3 accuracy.

In the current evaluation, Random Forest achieves an average rank of 2.333 and Top-1 accuracy of 50%. All three models achieve Top-3 accuracy of approximately 83.3%.

These results are based on six test seasons from 2020 to 2025 and should be interpreted in the context of the evaluation method used.

## Feature Importance

The Random Forest feature-importance analysis identifies points per game and minutes played per game as the two most influential features in the current model.

## Project Structure

```text
NBA-MVP-ML/
├── data/
│   ├── Player Award Shares.csv
│   └── Player Per Game.csv
├── results/
│   ├── model_comparison.png
│   └── feature_importance.png
├── NBA_MVP_Prediction.ipynb
├── requirements.txt
└── README.md
```

## How to Run

1. Install Python 3.11 or a compatible version.
2. Open a terminal in the project directory.
3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Open the notebook in Jupyter Notebook or Visual Studio Code.
5. Run all cells from top to bottom.

Ensure both CSV files are present in the `data/` directory before running the notebook.

## Results

The project generates two visualizations:

- `results/model_comparison.png` — compares model ranking accuracy.
- `results/feature_importance.png` — displays Random Forest feature importance.

## Limitations

- The evaluation covers only six test seasons.
- MVP voting is influenced by factors beyond basic per-game statistics.
- Historical patterns may not generalize to future seasons.
- Feature importance indicates predictive contribution, not causation.

## Data Sources

- Basketball-Reference: https://www.basketball-reference.com/
- Kaggle NBA/ABA/BAA statistics dataset: https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats

This project is intended for educational and experimental purposes.

