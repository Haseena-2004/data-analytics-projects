# Crop Recommendation - Predictive Modeling

Multi-class classification that recommends one of 22 crops from soil nutrients (N, P, K), temperature, humidity, pH and rainfall.

## Data
- **Default run: synthetic data** generated in `crop_recommendation.py` (22 crop profiles, 100 samples each, same schema as the public Kaggle *Crop Recommendation Dataset*).
- **To use the real dataset:** download `Crop_recommendation.csv` from Kaggle, place it in `data/`, and re-run. The script detects it automatically and prints which source it used.
- Results below are from the **synthetic** run. Real-data scores will differ (that dataset is much more separable), so re-run and update this table if you switch.

## Method
1. EDA: summary statistics and a correlation heatmap.
2. Stratified 80/20 train/test split (test set untouched until the end).
3. Baselines compared with 5-fold stratified CV on the training set: Logistic Regression, KNN, Random Forest.
4. `GridSearchCV` on Random Forest and Logistic Regression; the best CV model is selected.
5. Final evaluation once on the held-out test set; overfitting checked by comparing train vs test accuracy.

## Results (synthetic data, held-out test set, 440 samples)
| Metric | Score |
|---|---|
| Accuracy | 92.05% |
| Precision (macro) | 92.24% |
| Recall (macro) | 92.05% |
| F1 (macro) | 91.99% |

Selected model: Logistic Regression (C=10), 5-fold CV accuracy 92.3% vs Random Forest 90.7% and KNN 90.1%. Train accuracy 94.1% vs test 92.0%, so there is mild overfitting but no major gap.

Outputs are in `results/`: `metrics.json`, `confusion_matrix.png`, `feature_importance.png`, `correlation.png`.

## Limitations
- Synthetic features are drawn from per-crop Gaussians, so results show the pipeline works, not real agronomic performance.
- No geographic or seasonal effects are modeled.

## Run
```
pip install -r requirements.txt
python crop_recommendation.py
```
