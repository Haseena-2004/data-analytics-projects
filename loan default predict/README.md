# Loan Default Prediction - Financial Data Science

Predicts loan default from applicant and loan features, and compares three models using cross-validation.

## Data
**Synthetic** (6,000 loans, ~25.5% default rate) generated in `loan_default_model.py`; no real customer data is used. Default probability depends on credit score, debt-to-income, loan-to-income, employment length, prior delinquencies and loan purpose, plus noise. About 2-3% of two columns are set to missing to practise cleaning. The generated file is saved as `loan_data_synthetic.csv`.

## Method
1. Median imputation, scaling of numeric features, one-hot encoding of loan purpose.
2. Feature engineering: `loan_to_income`.
3. Stratified 80/20 split; class imbalance handled with `class_weight="balanced"` (Gradient Boosting left unweighted).
4. 5-fold stratified CV on the training set for Logistic Regression, Random Forest and Gradient Boosting (accuracy, precision, recall, F1, ROC-AUC).
5. Model chosen by CV ROC-AUC, then evaluated once on the held-out test set.
6. Precision/recall trade-off across decision thresholds.

## Results
5-fold CV (training set):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.673 | 0.412 | 0.667 | 0.509 | **0.737** |
| Random Forest | 0.737 | 0.483 | 0.455 | 0.468 | 0.720 |
| Gradient Boosting | 0.760 | 0.574 | 0.222 | 0.320 | 0.717 |

Held-out test set (Logistic Regression): accuracy 0.686, precision 0.426, recall 0.663, F1 0.519, ROC-AUC 0.730 (train AUC 0.741, so little overfitting).

Gradient Boosting has the highest accuracy but misses most defaulters (recall 0.22), which is why ROC-AUC and recall were used for selection rather than accuracy alone.

## Limitations
- Synthetic data with a noisy outcome, so a moderate AUC (~0.73) is expected; real-world scores will differ.
- No fairness, cost-sensitive threshold or calibration analysis.

## Run
```
pip install -r requirements.txt
python loan_default_model.py
```
