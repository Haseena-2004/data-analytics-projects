"""
Loan Default Prediction - financial data science and predictive modeling.

Uses a SYNTHETIC loan dataset generated in code (no real customer data). Default
probability is driven by income, loan size, credit score, debt-to-income,
employment length, prior delinquencies and loan purpose, plus noise.
Models are compared with stratified cross-validation, then the chosen model is
evaluated once on a held-out test set.
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
warnings.filterwarnings("ignore")
SEED = 42

def make_data(n=6000):
    r = np.random.default_rng(SEED)
    income = np.exp(r.normal(10.9, 0.5, n)).round(-2)                  # annual income
    credit = np.clip(r.normal(680, 70, n), 450, 850).round()
    loan = np.clip(income * r.uniform(0.1, 0.9, n), 1000, 60000).round(-2)
    dti = np.clip(r.normal(0.28, 0.1, n), 0.02, 0.7).round(3)
    emp = np.clip(r.exponential(5, n), 0, 35).round(1)
    delinq = r.poisson(0.35, n)
    purpose = r.choice(["debt_consolidation", "home_improvement", "education", "medical", "business"], n, p=[.4, .2, .15, .1, .15])
    z = (-2.2 - 0.012*(credit-680) + 3.0*(dti-0.28) + 1.6*(loan/income) - 0.04*emp + 0.45*delinq
         + np.where(purpose == "business", 0.5, 0) + r.normal(0, 0.6, n))
    default = (r.random(n) < 1/(1+np.exp(-z))).astype(int)
    df = pd.DataFrame(dict(annual_income=income, loan_amount=loan, credit_score=credit, debt_to_income=dti,
                           employment_years=emp, prior_delinquencies=delinq, purpose=purpose, default=default))
    # inject a little missingness so cleaning is realistic
    df.loc[r.random(n) < 0.03, "employment_years"] = np.nan
    df.loc[r.random(n) < 0.02, "debt_to_income"] = np.nan
    return df

df = make_data()
df.to_csv("loan_data_synthetic.csv", index=False)
print("Rows:", len(df), "| default rate:", round(df.default.mean(), 3))
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0].to_dict())

# feature engineering
df["loan_to_income"] = df.loan_amount / df.annual_income
num = ["annual_income", "loan_amount", "credit_score", "debt_to_income", "employment_years", "prior_delinquencies", "loan_to_income"]
cat = ["purpose"]
X, y = df[num + cat], df["default"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

from sklearn.impute import SimpleImputer
pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
models = {
 "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
 "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=SEED, n_jobs=-1),
 "Gradient Boosting": GradientBoostingClassifier(random_state=SEED)}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
rows = {}
for name, m in models.items():
    s = cross_validate(Pipeline([("pre", pre), ("m", m)]), Xtr, ytr, cv=cv, scoring=scoring)
    rows[name] = {k: round(float(s[f"test_{k}"].mean()), 4) for k in scoring}
cv_table = pd.DataFrame(rows).T
print("\n5-fold CV on training set:\n", cv_table)

best_name = cv_table["roc_auc"].idxmax()
best = Pipeline([("pre", pre), ("m", models[best_name])]).fit(Xtr, ytr)
proba = best.predict_proba(Xte)[:, 1]; pred = (proba >= 0.5).astype(int)
test = dict(accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred),
            recall=recall_score(yte, pred), f1=f1_score(yte, pred), roc_auc=roc_auc_score(yte, proba))
train_auc = roc_auc_score(ytr, best.predict_proba(Xtr)[:, 1])
print(f"\nSelected: {best_name}\nTest metrics:", {k: round(v, 4) for k, v in test.items()}, "| train AUC:", round(train_auc, 4))
print("Confusion matrix [[TN FP],[FN TP]]:", confusion_matrix(yte, pred).tolist())

# ROC curves for all models on the test set
plt.figure(figsize=(6, 5))
for name, m in models.items():
    p = Pipeline([("pre", pre), ("m", m)]).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, p); plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yte, p):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves - held-out test set"); plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig("results/roc_curves.png", dpi=130); plt.close()

# threshold trade-off for the selected model
ths = np.arange(0.1, 0.9, 0.05)
pr = [(precision_score(yte, proba >= t, zero_division=0), recall_score(yte, proba >= t)) for t in ths]
plt.figure(figsize=(6, 4)); plt.plot(ths, [a for a, _ in pr], label="precision"); plt.plot(ths, [b for _, b in pr], label="recall")
plt.xlabel("decision threshold"); plt.legend(); plt.title(f"Precision/recall trade-off ({best_name})"); plt.tight_layout()
plt.savefig("results/threshold_tradeoff.png", dpi=130); plt.close()

json.dump(dict(data="synthetic", rows=len(df), default_rate=float(df.default.mean()), cv_results=rows, selected_model=best_name,
               test_metrics=test, train_roc_auc=train_auc), open("results/metrics.json", "w"), indent=2)
