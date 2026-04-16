"""
backend/app.py  —  KisanCredit AI  Flask REST API
Integrates: ML prediction + chatbot blueprint
"""
import os, json, sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import joblib

# ── PATH SETUP ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))   # .../FarmerAI/backend
ROOT = os.path.join(BASE, "..")                      # .../FarmerAI
sys.path.insert(0, ROOT)                             # makes 'chatbot' package importable

MDIR = os.path.join(ROOT, "models")
ODIR = os.path.join(ROOT, "outputs")
DATA = os.path.join(ROOT, "data", "loan_train.csv")

# ── FLASK APP ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "kisancredit-dev-secret-change-in-prod")

# ── CORS ───────────────────────────────────────────────────────────────────────
@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return r

@app.route("/api/<path:p>", methods=["OPTIONS"])
def preflight(p):
    return "", 204

# ── REGISTER CHATBOT BLUEPRINT ─────────────────────────────────────────────────
try:
    from chatbot.api import chatbot_bp
    app.register_blueprint(chatbot_bp)
    print("[INFO] Chatbot blueprint registered → /api/chat/*")
except ImportError as e:
    print(f"[WARN] Chatbot blueprint not loaded: {e}")

# ── ML ARTEFACTS ───────────────────────────────────────────────────────────────
CAT_MAP = {
    "Gender":        {"Male": 1, "Female": 0},
    "Married":       {"Yes": 1, "No": 0},
    "Dependents":    {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education":     {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
}


def load_artefacts():
    try:
        return (
            joblib.load(os.path.join(MDIR, "best_model.pkl")),
            joblib.load(os.path.join(MDIR, "scaler.pkl")),
            joblib.load(os.path.join(MDIR, "pca.pkl")),
            joblib.load(os.path.join(MDIR, "label_encoders.pkl")),
            joblib.load(os.path.join(MDIR, "feature_cols.pkl")),
        )
    except Exception:
        return None, None, None, None, None


MODEL, SCALER, PCA_T, LE_DICT, FEAT_COLS = load_artefacts()


def build_row(data):
    d = dict(data)
    for col, mp in CAT_MAP.items():
        d[col] = mp.get(str(d.get(col, "")), 0)
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                "Loan_Amount_Term", "Credit_History"]:
        d[col] = float(d.get(col, 0) or 0)

    df = pd.DataFrame([d])
    df["TotalIncome"]        = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["TotalIncome_log"]    = np.log1p(df["TotalIncome"])
    df["LoanAmount_log"]     = np.log1p(df["LoanAmount"])
    df["EMI"]                = df["LoanAmount"] / (df["Loan_Amount_Term"] + 1e-9)
    df["BalanceIncome"]      = df["TotalIncome"] - df["EMI"] * 1000
    df["Debt_to_Income"]     = df["LoanAmount"] / (df["TotalIncome"] + 1)
    dep = float(str(d.get("Dependents", 0)).replace("3+", "3"))
    df["Income_per_Person"]  = df["TotalIncome"] / (dep + 2)
    df["LoanToIncome_ratio"] = df["LoanAmount"] / (df["ApplicantIncome"] + 1)

    for col, bins in [
        ("ApplicantIncome", [-np.inf, 2200, 3500, 5500, 9000, np.inf]),
        ("LoanAmount",      [-np.inf,   75,  110,  150,  220, np.inf]),
        ("TotalIncome",     [-np.inf, 2500, 4000, 6500, 10000, np.inf]),
    ]:
        df[f"{col}_bin"] = (
            pd.cut(df[col], bins=bins, labels=[0, 1, 2, 3, 4])
            .astype(float).fillna(2).astype(int)
        )

    if FEAT_COLS:
        for c in FEAT_COLS:
            if c not in df.columns:
                df[c] = 0
        return df[FEAT_COLS].fillna(0)
    return df.fillna(0)


def get_factors(data, prob):
    credit  = int(data.get("Credit_History", 1))
    inc     = float(data.get("ApplicantIncome", 0)) + float(data.get("CoapplicantIncome", 0))
    loan    = float(data.get("LoanAmount", 0))
    edu     = data.get("Education", "Graduate")
    married = data.get("Married", "No")
    land    = float(data.get("LandAcres", 0))
    facs    = []

    facs.append({
        "factor": "Good credit history" if credit == 1 else "Poor credit history",
        "impact": "+" if credit == 1 else "-",
        "weight": 0.38,
    })
    if inc > 6000:
        facs.append({"factor": f"Strong income ₹{inc:,.0f}", "impact": "+", "weight": 0.22})
    elif inc < 2500:
        facs.append({"factor": "Low combined income",          "impact": "-", "weight": 0.22})
    else:
        facs.append({"factor": "Moderate income level",        "impact": "~", "weight": 0.22})

    if loan < 130:
        facs.append({"factor": "Conservative loan amount",   "impact": "+", "weight": 0.15})
    elif loan > 400:
        facs.append({"factor": "High loan exposure",          "impact": "-", "weight": 0.15})

    if edu == "Graduate":
        facs.append({"factor": "Graduate education",          "impact": "+", "weight": 0.08})
    if married == "Yes":
        facs.append({"factor": "Married applicant",           "impact": "+", "weight": 0.05})
    if land > 5:
        facs.append({"factor": f"Large land holding ({land:.1f} ac)", "impact": "+", "weight": 0.07})

    return facs


# ── ML ROUTES ──────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": MODEL is not None,
        "model_type":   type(MODEL).__name__ if MODEL else None,
        "chatbot":      "ready",
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not ready. Call GET /api/train first."}), 503
    data = request.get_json(force=True) or {}
    try:
        row  = build_row(data)
        xs   = SCALER.transform(row)
        xp   = PCA_T.transform(xs)
        pred = int(MODEL.predict(xp)[0])
        prob = float(MODEL.predict_proba(xp)[0][1])

        if prob >= 0.75:
            rl, rc = "LOW RISK",    "#2d9e4e"
        elif prob >= 0.50:
            rl, rc = "MEDIUM RISK", "#e8960a"
        else:
            rl, rc = "HIGH RISK",   "#e03333"

        return jsonify({
            "approved":        pred == 1,
            "probability":     round(prob, 4),
            "risk_level":      rl,
            "risk_color":      rc,
            "risk_score":      round((1 - prob) * 100, 1),
            "recommendation":  ("Loan Approved — Low credit risk."
                                if pred == 1 else
                                "Loan Flagged — Manual review recommended."),
            "factors":         get_factors(data, prob),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/results")
def results():
    p = os.path.join(ODIR, "results_summary.json")
    if not os.path.exists(p):
        return jsonify({"error": "No results found. Run /api/train first."}), 404
    with open(p) as f:
        return jsonify(json.load(f))


@app.route("/api/train", methods=["GET", "POST"])
def train():
    try:
        sys.path.insert(0, BASE)
        from ml_pipeline import run_pipeline          # backend/ml_pipeline.py
        res, best = run_pipeline()
        global MODEL, SCALER, PCA_T, LE_DICT, FEAT_COLS
        MODEL, SCALER, PCA_T, LE_DICT, FEAT_COLS = load_artefacts()
        return jsonify({
            "status":     "complete",
            "best_model": best,
            "results": {
                k: {"accuracy": v["accuracy"], "roc_auc": v["roc_auc"]}
                for k, v in res.items()
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset/stats")
def dstats():
    if not os.path.exists(DATA):
        return jsonify({"error": "Dataset missing"}), 404
    df = pd.read_csv(DATA)
    for c in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
              "Loan_Amount_Term", "Credit_History"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return jsonify({
        "rows":          len(df),
        "columns":       list(df.columns),
        "approval_rate": round((df["Loan_Status"] == "Y").mean() * 100, 2),
        "missing":       df.isnull().sum().to_dict(),
    })


@app.route("/api/dataset/sample")
def dsample():
    if not os.path.exists(DATA):
        return jsonify({"error": "Dataset missing"}), 404
    n = min(int(request.args.get("n", 10)), 50)
    return jsonify(
        pd.read_csv(DATA).head(n).fillna("").to_dict(orient="records")
    )


# ── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("KisanCredit AI API  →  http://localhost:5000")
    print("Chatbot endpoint    →  POST http://localhost:5000/api/chat")
    app.run(host="0.0.0.0", port=5000, debug=True)
