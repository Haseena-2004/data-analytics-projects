"""
Crop Recommendation - predictive modeling with scikit-learn.

Data
----
If data/Crop_recommendation.csv exists (the public Kaggle "Crop Recommendation
Dataset": N, P, K, temperature, humidity, ph, rainfall, label) it is used.
Otherwise a synthetic dataset with the same schema is generated so the project
runs end to end. The data source used is printed and saved in results/metrics.json.
"""
import json, os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
warnings.filterwarnings("ignore")
SEED = 42
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CSV = "data/Crop_recommendation.csv"

# (N, P, K, temp, humidity, ph, rainfall) typical means per crop - synthetic profiles
PROFILES = {
 "rice": (80,48,40,24,82,6.4,236), "maize": (78,48,20,22,65,6.2,85), "chickpea": (40,68,80,19,17,7.3,80),
 "kidneybeans": (21,68,20,20,22,5.7,106), "pigeonpeas": (21,68,20,28,48,5.8,149), "mothbeans": (21,48,20,28,53,6.8,51),
 "mungbean": (21,48,20,28,85,6.7,48), "blackgram": (40,68,19,30,65,7.1,68), "lentil": (19,68,19,25,65,6.9,46),
 "pomegranate": (19,19,40,22,90,6.4,108), "banana": (100,82,50,27,80,6.0,105), "mango": (20,28,30,31,50,5.8,95),
 "grapes": (23,132,200,24,82,6.0,70), "watermelon": (99,17,50,25,85,6.5,50), "muskmelon": (100,18,50,28,92,6.4,25),
 "apple": (21,134,200,22,92,5.9,113), "orange": (20,17,10,23,92,7.0,110), "papaya": (50,59,50,34,92,6.7,143),
 "coconut": (22,17,31,27,95,6.0,176), "cotton": (118,46,20,24,80,6.9,80), "jute": (78,46,40,25,80,6.7,175),
 "coffee": (101,29,30,25,58,6.8,158)}

def synthetic(n_per_class=100):
    rng = np.random.default_rng(SEED); rows = []
    for crop, p in PROFILES.items():
        sd = np.array([12, 10, 10, 2.5, 6, 0.45, 22])      # within-crop spread
        x = rng.normal(p, sd, size=(n_per_class, 7))
        x = np.clip(x, [0,5,5,8,14,3.5,20], [140,145,205,44,100,9.9,300])
        d = pd.DataFrame(x, columns=FEATURES); d["label"] = crop; rows.append(d)
    return pd.concat(rows, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

if os.path.exists(CSV):
    df, source = pd.read_csv(CSV), "Kaggle Crop Recommendation Dataset (data/Crop_recommendation.csv)"
else:
    df, source = synthetic(), "SYNTHETIC data generated in code (same schema as the Kaggle dataset)"
print("Data source:", source, "| shape:", df.shape)
assert df.isna().sum().sum() == 0

# --- EDA ---
print(df.describe().round(2).T[["mean", "std", "min", "max"]])
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(df[FEATURES].corr(), cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(7)); ax.set_xticklabels(FEATURES, rotation=45, ha="right"); ax.set_yticks(range(7)); ax.set_yticklabels(FEATURES)
plt.colorbar(im); ax.set_title("Feature correlation"); plt.tight_layout(); plt.savefig("results/correlation.png", dpi=130); plt.close()

# --- Split + baseline comparison with CV (train only) ---
X, y = df[FEATURES], df["label"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
models = {
 "Logistic Regression": Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=2000))]),
 "KNN": Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier())]),
 "Random Forest": RandomForestClassifier(random_state=SEED)}
cv_scores = {k: cross_val_score(m, Xtr, ytr, cv=cv, scoring="accuracy").mean() for k, m in models.items()}
print("5-fold CV accuracy (train set):", {k: round(v, 4) for k, v in cv_scores.items()})

# --- Tune candidate models with GridSearchCV, pick the best by CV score ---
searches = {
 "Random Forest": GridSearchCV(RandomForestClassifier(random_state=SEED),
    {"n_estimators": [100, 300], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 2]},
    cv=cv, scoring="accuracy", n_jobs=-1),
 "Logistic Regression": GridSearchCV(models["Logistic Regression"],
    {"m__C": [0.1, 1, 10, 100]}, cv=cv, scoring="accuracy", n_jobs=-1)}
for name, g in searches.items():
    g.fit(Xtr, ytr); print(f"{name}: best CV acc {g.best_score_:.4f} | {g.best_params_}")
best_name = max(searches, key=lambda k: searches[k].best_score_)
grid = searches[best_name]; best = grid.best_estimator_
print("Selected model:", best_name)

# --- Final evaluation on untouched test set ---
pred = best.predict(Xte)
m = dict(accuracy=accuracy_score(yte, pred),
         precision_macro=precision_score(yte, pred, average="macro"),
         recall_macro=recall_score(yte, pred, average="macro"),
         f1_macro=f1_score(yte, pred, average="macro"))
train_acc = accuracy_score(ytr, best.predict(Xtr))
print({k: round(v, 4) for k, v in m.items()}, "| train acc:", round(train_acc, 4))

fig, ax = plt.subplots(figsize=(10, 9))
ConfusionMatrixDisplay(confusion_matrix(yte, pred, labels=best.classes_), display_labels=best.classes_).plot(ax=ax, xticks_rotation=90, colorbar=False)
plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=110); plt.close()
rf = searches["Random Forest"].best_estimator_
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
imp.plot.barh(title="Random Forest feature importance", figsize=(6, 4)); plt.tight_layout(); plt.savefig("results/feature_importance.png", dpi=130); plt.close()

json.dump(dict(data_source=source, rows=len(df), selected_model=best_name, cv_accuracy=cv_scores, best_params=grid.best_params_,
               test_metrics=m, train_accuracy=train_acc, feature_importance=imp.to_dict()),
          open("results/metrics.json", "w"), indent=2)
