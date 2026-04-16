"""
backend/ml_pipeline.py
ML training pipeline for KisanCredit AI.
Run directly:  python backend/ml_pipeline.py
Or via API:    GET/POST http://localhost:5000/api/train
"""
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing  import LabelEncoder, StandardScaler, KBinsDiscretizer
from sklearn.decomposition   import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (accuracy_score, classification_report,
                                    confusion_matrix, roc_auc_score, roc_curve,
                                    average_precision_score, precision_recall_curve)
import joblib

warnings.filterwarnings("ignore")

# Paths — all relative to FarmerAI/ root
BASE = os.path.dirname(os.path.abspath(__file__))   # .../FarmerAI/backend
ROOT = os.path.join(BASE, "..")                      # .../FarmerAI
DATA = os.path.join(ROOT, "data")
MDL  = os.path.join(ROOT, "models")
OUT  = os.path.join(ROOT, "outputs")
CSV  = os.path.join(DATA, "loan_train.csv")

for d in [DATA, MDL, OUT]:
    os.makedirs(d, exist_ok=True)

PAL = dict(g="#34a853", a="#fbbc04", b="#1a8cff", r="#e03333",
           bg="#0a1a0f", txt="#e2f2e6")


def _ax(ax):
    ax.set_facecolor(PAL["bg"])
    for sp in ax.spines.values():
        sp.set_edgecolor("#1b3a22")


# ── STEP 0 ─────────────────────────────────────────────────────────────────────
def load():
    if not os.path.exists(CSV):
        raise FileNotFoundError(
            f"\nDataset not found: {CSV}\n"
            "Download from:\n"
            "  kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset\n"
            "Save as:  FarmerAI/data/loan_train.csv\n"
            "Kaggle CLI:\n"
            "  kaggle datasets download -d altruistdelhite04/"
            "loan-prediction-problem-dataset --unzip -p data/"
        )
    df = pd.read_csv(CSV)
    print(f"[STEP 0] Loaded {df.shape[0]} rows × {df.shape[1]} cols  ← data/loan_train.csv")
    print(f"         Columns: {list(df.columns)}")
    return df


# ── STEP 1 ─────────────────────────────────────────────────────────────────────
def eda(df):
    print("\n[STEP 1] EDA")
    miss = df.isnull().sum()
    print(f"  Missing:\n{miss[miss > 0].to_string()}")
    print(f"  Target:  {df['Loan_Status'].value_counts().to_dict()}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=PAL["bg"])
    plt.suptitle("KisanCredit AI — Kaggle Loan Prediction EDA",
                 color=PAL["txt"], fontsize=13)
    panels = [
        ("Loan Status",   df["Loan_Status"].value_counts(),   [PAL["g"], PAL["r"]]),
        ("Property Area", df["Property_Area"].value_counts(), [PAL["b"], PAL["a"], PAL["g"]]),
        ("Education",     df["Education"].value_counts(),     [PAL["g"], PAL["a"]]),
        ("Approval% by Credit History",
         df.groupby("Credit_History")["Loan_Status"].apply(
             lambda x: (x == "Y").mean() * 100),
         [PAL["r"], PAL["g"]]),
    ]
    for ax, (title, data, colors) in zip(axes.flat, panels):
        _ax(ax)
        ax.bar(data.index.astype(str), data.values,
               color=colors[:len(data)], edgecolor="none", width=0.45)
        ax.set_title(title, color=PAL["txt"], fontsize=11)
        ax.tick_params(colors=PAL["txt"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "eda_overview.png"),
                dpi=150, bbox_inches="tight", facecolor=PAL["bg"])
    plt.close()

    fig2, ax2s = plt.subplots(1, 2, figsize=(10, 4), facecolor=PAL["bg"])
    for ax, col, color in zip(ax2s,
                               ["ApplicantIncome", "LoanAmount"],
                               [PAL["g"], PAL["a"]]):
        _ax(ax)
        ax.hist(df[col].dropna(), bins=40, color=color, edgecolor="none", alpha=0.8)
        ax.set_title(f"{col} Distribution", color=PAL["txt"])
        ax.tick_params(colors=PAL["txt"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "income_distribution.png"),
                dpi=150, bbox_inches="tight", facecolor=PAL["bg"])
    plt.close()
    print("  Saved: eda_overview.png, income_distribution.png")


# ── STEP 2 ─────────────────────────────────────────────────────────────────────
def clean(df):
    print("\n[STEP 2] Cleaning")
    df = df.drop(columns=["Loan_ID", "Applicant_Name"], errors="ignore").copy()
    for c in ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mode()[0])
    for c in ["LoanAmount", "Loan_Amount_Term"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    df = df.drop_duplicates()
    print(f"  NaN remaining: {df.isnull().sum().sum()} | Shape: {df.shape}")
    return df


# ── STEP 3 ─────────────────────────────────────────────────────────────────────
def engineer(df):
    print("\n[STEP 3] Feature Engineering (8 new features)")
    df = df.copy()
    df["TotalIncome"]        = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["TotalIncome_log"]    = np.log1p(df["TotalIncome"])
    df["LoanAmount_log"]     = np.log1p(df["LoanAmount"].fillna(0))
    df["EMI"]                = df["LoanAmount"] / (df["Loan_Amount_Term"] + 1e-9)
    df["BalanceIncome"]      = df["TotalIncome"] - (df["EMI"] * 1000)
    df["Debt_to_Income"]     = df["LoanAmount"] / (df["TotalIncome"] + 1)
    df["Income_per_Person"]  = df["TotalIncome"] / (
        df["Dependents"].replace({"3+": 4}).astype(float) + 2)
    df["LoanToIncome_ratio"] = df["LoanAmount"] / (df["ApplicantIncome"] + 1)
    print("  Added: TotalIncome, TotalIncome_log, LoanAmount_log, EMI,")
    print("         BalanceIncome, Debt_to_Income, Income_per_Person, LoanToIncome_ratio")
    return df


# ── STEP 4 ─────────────────────────────────────────────────────────────────────
def encode(df):
    print("\n[STEP 4] Label Encoding")
    df = df.copy()
    le_map = {}
    cats = ["Gender", "Married", "Dependents", "Education",
            "Self_Employed", "Property_Area", "Loan_Status"]
    for c in cats:
        if c in df.columns:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            le_map[c] = le
    print(f"  Encoded: {[c for c in cats if c in df.columns]}")
    return df, le_map


# ── STEP 5 ─────────────────────────────────────────────────────────────────────
def discretise(df):
    print("\n[STEP 5] KBins Discretisation (quantile, n=5)")
    df = df.copy()
    for c in ["ApplicantIncome", "LoanAmount", "TotalIncome"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
            kbd = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
            df[f"{c}_bin"] = kbd.fit_transform(df[[c]]).astype(int)
    return df


# ── STEP 6 ─────────────────────────────────────────────────────────────────────
def normalise(X):
    print("\n[STEP 6] StandardScaler")
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    print(f"  Feature matrix: {Xs.shape}")
    return Xs, sc


# ── STEP 7 ─────────────────────────────────────────────────────────────────────
def apply_pca(Xs, thresh=0.95):
    print("\n[STEP 7] PCA")
    full   = PCA().fit(Xs)
    cumvar = np.cumsum(full.explained_variance_ratio_)
    n      = int(np.argmax(cumvar >= thresh) + 1)
    print(f"  {n} components for {thresh * 100:.0f}% variance")
    pca = PCA(n_components=n)
    Xp  = pca.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=PAL["bg"])
    _ax(ax)
    ax.plot(range(1, len(cumvar) + 1), cumvar,
            color=PAL["g"], lw=2, marker="o", markersize=4)
    ax.axhline(thresh, color=PAL["a"], ls="--", lw=1.5,
               label=f"{thresh * 100:.0f}% threshold")
    ax.axvline(n, color=PAL["b"], ls="--", lw=1.5, label=f"n={n}")
    ax.set_xlabel("Components", color=PAL["txt"])
    ax.set_ylabel("Cumulative Variance", color=PAL["txt"])
    ax.set_title("PCA Cumulative Explained Variance", color=PAL["txt"])
    ax.tick_params(colors=PAL["txt"])
    ax.legend(labelcolor=PAL["txt"], facecolor=PAL["bg"], edgecolor="none")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "pca_variance.png"),
                dpi=150, bbox_inches="tight", facecolor=PAL["bg"])
    plt.close()
    print(f"  PCA output: {Xp.shape}")
    return Xp, pca


# ── STEP 8 ─────────────────────────────────────────────────────────────────────
def split(X, y):
    print("\n[STEP 8] Stratified 80/20 Split")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {Xtr.shape}  |  Test: {Xte.shape}")
    return Xtr, Xte, ytr, yte


# ── STEP 9 ─────────────────────────────────────────────────────────────────────
def train_models(Xtr, ytr):
    print("\n[STEP 9] Training + 5-Fold CV")
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=4,
            subsample=0.8, random_state=42),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, C=0.8, class_weight="balanced", random_state=42),
    }
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trained = {}
    cv_res  = {}
    for name, m in models.items():
        scores = cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)
        m.fit(Xtr, ytr)
        trained[name] = m
        cv_res[name]  = scores
        print(f"  {name:<25}  CV AUC = {scores.mean():.4f} ± {scores.std():.4f}")
    return trained, cv_res


# ── STEP 10 ────────────────────────────────────────────────────────────────────
def evaluate(trained, Xte, yte):
    print("\n[STEP 10] Evaluation")
    results = {}
    colors  = [PAL["g"], PAL["b"], PAL["a"]]

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6), facecolor=PAL["bg"])
    _ax(ax_roc); ax_roc.tick_params(colors=PAL["txt"])
    fig_pr, ax_pr   = plt.subplots(figsize=(8, 6), facecolor=PAL["bg"])
    _ax(ax_pr);  ax_pr.tick_params(colors=PAL["txt"])

    for (name, m), col in zip(trained.items(), colors):
        yp     = m.predict(Xte)
        prob   = m.predict_proba(Xte)[:, 1]
        acc    = accuracy_score(yte, yp)
        auc    = roc_auc_score(yte, prob)
        pr_auc = average_precision_score(yte, prob)
        cm     = confusion_matrix(yte, yp)
        cr     = classification_report(yte, yp, output_dict=True)
        fpr, tpr, _   = roc_curve(yte, prob)
        prec, rec, _  = precision_recall_curve(yte, prob)

        results[name] = {
            "accuracy":               round(acc,    4),
            "roc_auc":                round(auc,    4),
            "pr_auc":                 round(pr_auc, 4),
            "confusion_matrix":       cm.tolist(),
            "classification_report":  cr,
        }
        ax_roc.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={auc:.3f})")
        ax_pr.plot(rec, prec, color=col, lw=2, label=f"{name} (PR={pr_auc:.3f})")
        print(f"  {name:<25}  Acc={acc:.4f}  AUC={auc:.4f}  PR={pr_auc:.4f}")
        print(classification_report(yte, yp, target_names=["Rejected", "Approved"]))

    ax_roc.plot([0, 1], [0, 1], "--", color=(1, 1, 1, 0.12), lw=1)
    ax_roc.set_xlabel("FPR", color=PAL["txt"])
    ax_roc.set_ylabel("TPR", color=PAL["txt"])
    ax_roc.set_title("ROC-AUC Curves", color=PAL["txt"])
    ax_roc.legend(labelcolor=PAL["txt"], facecolor=PAL["bg"], edgecolor="none")
    plt.figure(fig_roc.number); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "roc_auc.png"),
                dpi=150, bbox_inches="tight", facecolor=PAL["bg"])
    plt.close(fig_roc)

    ax_pr.set_xlabel("Recall", color=PAL["txt"])
    ax_pr.set_ylabel("Precision", color=PAL["txt"])
    ax_pr.set_title("Precision-Recall Curves", color=PAL["txt"])
    ax_pr.legend(labelcolor=PAL["txt"], facecolor=PAL["bg"], edgecolor="none")
    plt.figure(fig_pr.number); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "pr_curve.png"),
                dpi=150, bbox_inches="tight", facecolor=PAL["bg"])
    plt.close(fig_pr)

    best = max(results, key=lambda k: results[k]["roc_auc"])
    bcm  = np.array(results[best]["confusion_matrix"])
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4), facecolor=PAL["bg"])
    ax_cm.set_facecolor(PAL["bg"])
    sns.heatmap(bcm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"],
                ax=ax_cm, linewidths=0.5, annot_kws={"size": 14, "weight": "bold"})
    ax_cm.set_title(f"Confusion Matrix — {best}", color=PAL["txt"])
    ax_cm.tick_params(colors=PAL["txt"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "confusion_matrix.png"),
                dpi=150, bbox_inches="tight", facecolor=PAL["bg"])
    plt.close()
    print(f"\n  Best model: {best}  (AUC={results[best]['roc_auc']})")
    return results, best


def feat_importance(model, n_feats):
    if not hasattr(model, "feature_importances_"):
        return
    imp = (pd.Series(model.feature_importances_,
                     index=[f"PC{i + 1}" for i in range(n_feats)])
           .sort_values(ascending=True).tail(12))
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=PAL["bg"])
    _ax(ax); ax.tick_params(colors=PAL["txt"])
    cols = [PAL["a"] if i == len(imp) - 1 else PAL["g"] for i in range(len(imp))]
    ax.barh(imp.index, imp.values, color=cols, edgecolor="none", height=0.55)
    ax.set_xlabel("Importance", color=PAL["txt"])
    ax.set_title("Feature Importance (PCA Components)", color=PAL["txt"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "feature_importance.png"),
                dpi=150, bbox_inches="tight", facecolor=PAL["bg"])
    plt.close()
    print("  Saved: feature_importance.png")


def corr_heatmap(df_enc):
    num  = df_enc.select_dtypes(include=np.number)
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(13, 10), facecolor=PAL["bg"])
    ax.set_facecolor(PAL["bg"])
    sns.heatmap(corr, cmap="RdYlGn", center=0, linewidths=0.3,
                ax=ax, annot=False, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", color=PAL["txt"], pad=12)
    ax.tick_params(colors=PAL["txt"], labelsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "correlation_heatmap.png"),
                dpi=150, bbox_inches="tight", facecolor=PAL["bg"])
    plt.close()
    print("  Saved: correlation_heatmap.png")


# ── STEP 11 ────────────────────────────────────────────────────────────────────
def save_artefacts(trained, scaler, pca, le_map, feat_cols, results, best):
    joblib.dump(trained[best],  os.path.join(MDL, "best_model.pkl"))
    joblib.dump(scaler,         os.path.join(MDL, "scaler.pkl"))
    joblib.dump(pca,            os.path.join(MDL, "pca.pkl"))
    joblib.dump(le_map,         os.path.join(MDL, "label_encoders.pkl"))
    joblib.dump(feat_cols,      os.path.join(MDL, "feature_cols.pkl"))

    summary = {
        "best_model":     best,
        "dataset_rows":   614,
        "dataset_source": "Kaggle Loan Prediction Problem",
        "dataset_path":   "data/loan_train.csv",
        "results": {
            k: {kk: vv for kk, vv in v.items()
                if kk != "classification_report"}
            for k, v in results.items()
        },
    }
    with open(os.path.join(OUT, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[STEP 11] Artefacts → models/ | Summary → outputs/results_summary.json")


# ── MAIN ───────────────────────────────────────────────────────────────────────
def run_pipeline():
    SEP = "=" * 64
    print(SEP)
    print("  KisanCredit AI  —  FARMER LOAN RISK PIPELINE")
    print("  Dataset: Kaggle Loan Prediction (614 rows, 14 cols)")
    print(SEP)

    df              = load()
    eda(df)
    df              = clean(df)
    df              = engineer(df)
    df, le_map      = encode(df)
    df              = discretise(df)
    corr_heatmap(df)

    target    = "Loan_Status"
    feat_cols = [c for c in df.columns if c != target]
    X, y      = df[feat_cols], df[target]
    print(f"\n  Total features: {len(feat_cols)} | Class split: {dict(y.value_counts())}")

    Xs, scaler        = normalise(X)
    Xp, pca           = apply_pca(Xs)
    Xtr, Xte, ytr, yte = split(Xp, y)
    trained, _        = train_models(Xtr, ytr)
    results, best     = evaluate(trained, Xte, yte)
    feat_importance(trained[best], Xp.shape[1])
    save_artefacts(trained, scaler, pca, le_map, feat_cols, results, best)

    print("\n" + SEP)
    print("  PIPELINE COMPLETE")
    print(SEP)
    return results, best


if __name__ == "__main__":
    run_pipeline()
