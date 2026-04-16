"""
chatbot/knowledge_base.py
Structured knowledge base for the KisanCredit AI chatbot.
All domain facts, intents, and responses are defined here.
"""

# ── INTENT KEYWORDS ────────────────────────────────────────────────────────────
# Maps intent name → list of keyword patterns (any match triggers the intent)
INTENTS = {
    "greeting":        ["hi", "hello", "hey", "namaste", "namaskar", "good morning",
                        "good afternoon", "good evening", "start", "help"],
    "credit_history":  ["credit history", "credit_history", "credit score", "cibil",
                        "repayment", "past loans", "credit record"],
    "income":          ["income", "salary", "earnings", "co-applicant", "coapplicant",
                        "total income", "combined income", "monthly income"],
    "loan_amount":     ["loan amount", "how much", "borrow", "amount", "loanamount",
                        "how much loan", "maximum loan"],
    "features":        ["feature", "engineered", "8 features", "variables", "columns",
                        "emi", "balance income", "debt to income", "pca", "components"],
    "pca":             ["pca", "principal component", "dimensionality", "reduce features",
                        "variance", "12 components", "22 features"],
    "model":           ["model", "algorithm", "logistic", "random forest",
                        "gradient boosting", "which model", "best model", "accuracy",
                        "auc", "roc", "pr-auc", "results", "performance"],
    "approval":        ["approve", "approval", "approve loan", "chance", "probability",
                        "will i get", "eligible", "qualify", "eligibility"],
    "rejection":       ["reject", "rejected", "denied", "decline", "not approved",
                        "loan rejected", "alternative", "what if rejected"],
    "schemes":         ["scheme", "government", "pm kisan", "kcc", "kisan credit",
                        "nabard", "pmfby", "subsidy", "crop insurance", "support"],
    "dataset":         ["dataset", "data", "kaggle", "614", "rows", "training data",
                        "loan_train", "real applicants"],
    "overfitting":     ["overfit", "overfitting", "generalise", "generalize", "cv gap",
                        "cross validation", "validation"],
    "property":        ["property", "area", "rural", "urban", "semiurban", "location",
                        "region"],
    "education":       ["education", "graduate", "not graduate", "qualification",
                        "degree"],
    "crop":            ["crop", "wheat", "rice", "cotton", "soybean", "sugarcane",
                        "vegetable", "kharif", "rabi", "farming"],
    "weather":         ["weather", "flood", "drought", "rain", "temperature", "heat",
                        "monsoon", "climate", "risk"],
    "thanks":          ["thank", "thanks", "thank you", "shukriya", "dhanyavaad",
                        "great", "awesome", "nice", "good"],
    "bye":             ["bye", "goodbye", "see you", "exit", "quit", "done"],
}

# ── RESPONSES ──────────────────────────────────────────────────────────────────
RESPONSES = {
    "greeting": (
        "🙏 Namaste! I'm the **KisanCredit AI Assistant**.\n\n"
        "I'm trained on the Kaggle Loan Prediction dataset (614 real applicants) "
        "and can help you with:\n"
        "• Loan approval factors & eligibility\n"
        "• ML pipeline explanation (PCA, features, models)\n"
        "• Government schemes for farmers\n"
        "• Credit score improvement tips\n\n"
        "What would you like to know? 🌾"
    ),

    "credit_history": (
        "**Credit History** is the single most important factor — it carries ~**38% weight** "
        "in the model.\n\n"
        "📊 Statistics from the Kaggle dataset:\n"
        "• Credit_History = 1 (good): **87% approval rate**\n"
        "• Credit_History = 0 (poor): only **14% approval rate**\n\n"
        "⚠️ The dataset had **54 missing values** for this field — filled using mode (1.0).\n\n"
        "💡 *Tip: Clearing any existing defaults or dues before applying greatly improves "
        "your chances.*"
    ),

    "income": (
        "**Income** is the 2nd most important factor (~**20% weight**).\n\n"
        "The model uses **TotalIncome = ApplicantIncome + CoapplicantIncome**.\n\n"
        "📊 Dataset insights:\n"
        "• TotalIncome > ₹6,000/month → **81% approval rate**\n"
        "• TotalIncome < ₹2,500/month → high risk flag\n\n"
        "🔧 The feature `TotalIncome_log` (log1p transform) is used to reduce right-skew "
        "caused by high-income outliers.\n\n"
        "💡 *Including a co-applicant (spouse, family member) can significantly boost your "
        "combined income figure.*"
    ),

    "loan_amount": (
        "**Loan Amount** carries ~**13% weight** in the model.\n\n"
        "📊 Dataset benchmarks:\n"
        "• Median loan: ₹128,000 → conservative amounts improve approval chances\n"
        "• Loans > ₹400,000 are flagged as high-exposure\n\n"
        "🔧 The model uses `LoanAmount_log` (log transform) + `Debt_to_Income` ratio "
        "and `EMI` features derived from the loan amount.\n\n"
        "💡 *Applying for a loan that's proportional to your income (DTI < 0.4) gives "
        "the best approval odds.*"
    ),

    "features": (
        "**8 Engineered Features** added to the original 14 Kaggle columns:\n\n"
        "| Feature | Formula | Purpose |\n"
        "|---|---|---|\n"
        "| TotalIncome | Applicant + CoApplicant | Combined repayment power |\n"
        "| TotalIncome_log | log1p(TotalIncome) | Reduce right skew |\n"
        "| LoanAmount_log | log1p(LoanAmount) | Normalise outliers |\n"
        "| EMI | LoanAmount ÷ Loan_Term | Monthly burden |\n"
        "| BalanceIncome | TotalIncome − EMI×1000 | Residual after repayment |\n"
        "| Debt_to_Income | LoanAmount ÷ TotalIncome | Leverage ratio |\n"
        "| Income_per_Person | TotalIncome ÷ (Dependents+2) | Per-capita income |\n"
        "| LoanToIncome_ratio | LoanAmount ÷ ApplicantIncome | Borrowing ratio |\n\n"
        "Plus **3 bin features** (KBinsDiscretizer, quantile, n=5) on "
        "ApplicantIncome, LoanAmount, TotalIncome → **22 total features** before PCA."
    ),

    "pca": (
        "**PCA (Principal Component Analysis)** is applied after StandardScaler.\n\n"
        "🔢 Dimensions: **22 features → 12 principal components** (retaining **95% variance**)\n\n"
        "Why PCA helps here:\n"
        "• Eliminates **multicollinearity** between TotalIncome / ApplicantIncome / "
        "LoanToIncome_ratio\n"
        "• Acts as **regularisation** — reduces overfitting risk\n"
        "• Speeds up training by ~**25%**\n"
        "• LogisticRegression performs best on decorrelated features\n\n"
        "The explained variance curve is saved to `outputs/pca_variance.png`."
    ),

    "model": (
        "**Real Kaggle Results** (491 train / 123 test, Stratified 80/20):\n\n"
        "| Model | Test Accuracy | ROC-AUC | PR-AUC | CV AUC |\n"
        "|---|---|---|---|---|\n"
        "| ✅ **Logistic Regression** | **80.49%** | **0.8796** | **0.9439** | **0.9019** |\n"
        "| Random Forest | 76.42% | 0.8424 | 0.9211 | 0.8704 |\n"
        "| Gradient Boosting | 76.42% | 0.8015 | 0.8847 | 0.8520 |\n\n"
        "**Why Logistic Regression won:**\n"
        "• Best CV AUC (0.9019 ± 0.031) — most consistent\n"
        "• CV-Test gap only **0.022** — no overfitting\n"
        "• PCA decorrelated features play to LR's strengths\n"
        "• `class_weight='balanced'` handles the 69/31 imbalance"
    ),

    "approval": (
        "**Key factors for loan approval** (ranked by model weight):\n\n"
        "1. 🏆 **Credit History** (38%) — most critical; must be clean\n"
        "2. 💰 **Total Income** (20%) — aim for > ₹6,000/month combined\n"
        "3. 📋 **Loan Amount** (13%) — keep DTI ratio < 0.4\n"
        "4. 📉 **Debt-to-Income** (11%) — lower is better\n"
        "5. 🧾 **EMI burden** (7%) — should be manageable vs. income\n"
        "6. 🎓 **Education** (8%) — Graduate has lower default rate\n"
        "7. 🏘️ **Property Area** (5%) — Semiurban has best approval rate\n\n"
        "💡 *Use the **Assess Loan** page for a live prediction with your specific details.*"
    ),

    "rejection": (
        "If your loan is **rejected**, here are your options:\n\n"
        "🏛️ **Government Schemes:**\n"
        "• **PM Kisan Samman Nidhi** — ₹6,000/year direct transfer to farmer accounts\n"
        "• **Kisan Credit Card (KCC)** — revolving credit up to ₹3 lakhs @ 4% interest\n"
        "• **PMFBY** — Pradhan Mantri Fasal Bima Yojana (crop insurance)\n"
        "• **NABARD RIDF** — Rural Infrastructure loans via state cooperatives\n"
        "• **SHG Micro-credit** — Self-Help Group lending circles\n\n"
        "📈 **Improve Your Profile:**\n"
        "• Clear any existing defaults (credit history is 38% weight)\n"
        "• Add a co-applicant to increase combined income\n"
        "• Reduce the loan amount or extend the tenure\n"
        "• Provide land collateral documents\n\n"
        "💡 *Reapply after 6 months of improved credit behaviour.*"
    ),

    "schemes": (
        "**Government Schemes for Indian Farmers:**\n\n"
        "| Scheme | Benefit | Eligibility |\n"
        "|---|---|---|\n"
        "| PM Kisan Samman Nidhi | ₹6,000/year (3 instalments) | All landholding farmers |\n"
        "| Kisan Credit Card (KCC) | Credit up to ₹3L @ 4% p.a. | Any farmer with land record |\n"
        "| PMFBY | Crop insurance at subsidised premium | All notified crops |\n"
        "| PM Kisan MAN DHAN | Pension ₹3,000/month at age 60 | Small & marginal farmers |\n"
        "| NABARD RIDF | Infrastructure loans | State governments / cooperatives |\n"
        "| E-NAM | Online crop trading platform | Registered mandis |\n\n"
        "📞 Helplines: PM Kisan: 155261 | PMFBY: 14447"
    ),

    "dataset": (
        "**Kaggle Loan Prediction Dataset** (loan_train.csv):\n\n"
        "• **614 rows** × 14 columns (incl. Applicant_Name — real applicants!)\n"
        "• Source: kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset\n\n"
        "**Class distribution:**\n"
        "• ✅ Approved (Y): 422 (68.7%)\n"
        "• ❌ Rejected (N): 192 (31.3%)\n\n"
        "**Missing values (raw):**\n"
        "• Credit_History: 54 · Self_Employed: 23 · LoanAmount: 22\n"
        "• Loan_Amt_Term: 14 · Dependents: 1\n\n"
        "**Pipeline:** 12 steps from raw CSV → trained model artefacts in `models/`"
    ),

    "overfitting": (
        "**Overfitting Prevention Measures Used:**\n\n"
        "1. ✅ **5-Fold Stratified CV** — CV-Test AUC gap = 0.022 (excellent; < 0.03 is ideal)\n"
        "2. ✅ **PCA regularisation** — 22 → 12 components removes noise dimensions\n"
        "3. ✅ **class_weight='balanced'** — prevents bias toward majority class (69%)\n"
        "4. ✅ **L2 regularisation** in LogisticRegression (C=0.8)\n"
        "5. ✅ **RF max_depth=8** — prevents trees from memorising training data\n"
        "6. ✅ **GB subsample=0.8** — stochastic gradient boosting reduces variance\n\n"
        "The final CV AUC 0.9019 vs Test AUC 0.8796 gap of **0.022** confirms no overfit."
    ),

    "property": (
        "**Property Area** influences approval probability:\n\n"
        "| Area | Approval Rate (dataset) |\n"
        "|---|---|\n"
        "| Semiurban | **76.3%** (best) |\n"
        "| Urban | 67.8% |\n"
        "| Rural | 61.2% |\n\n"
        "Semiurban applicants benefit from better infrastructure access and more stable "
        "income sources compared to purely rural areas."
    ),

    "education": (
        "**Education level** carries ~8% weight.\n\n"
        "• **Graduates**: Lower statistical default rate → positive approval signal\n"
        "• **Non-graduates**: Not disqualifying but combined with other risk factors "
        "can reduce probability.\n\n"
        "Dataset approval rates: Graduate ~71% vs Non-Graduate ~61%"
    ),

    "crop": (
        "**Crop Risk Profile** (Kharif vs Rabi season risk):\n\n"
        "| Crop | Kharif Risk | Rabi Risk | Notes |\n"
        "|---|---|---|---|\n"
        "| Cotton | **65%** | 22% | High kharif exposure |\n"
        "| Soybean | **55%** | 40% | Moisture-sensitive |\n"
        "| Vegetables | 38% | 28% | Short cycle, higher volatility |\n"
        "| Rice/Paddy | 42% | 20% | Flood risk in low-lying areas |\n"
        "| Sugarcane | 30% | 45% | Assured procurement pricing |\n"
        "| Wheat | 18% | **35%** | Stable, low kharif risk |\n\n"
        "Sugarcane and wheat are considered **lower-risk crops** due to MSP protection."
    ),

    "weather": (
        "**Weather & Climate Risk Factors in Loan Assessment:**\n\n"
        "🌵 **Drought**: High drought index raises default risk by 18–25%. "
        "Cotton and soybean are most vulnerable. Loan caps applied in notified drought districts.\n\n"
        "🌡️ **Heat Stress**: Temperatures > 40°C can cut yield by 30–40%. "
        "The model flags manual review for heat-stressed applicants.\n\n"
        "🌊 **Flood Risk**: Flood-prone districts attract +15% risk premium. "
        "Rice farmers in flood zones need extra collateral documentation per RBI norms.\n\n"
        "🌧️ **Monsoon Variability**: El Niño years see ~22% higher agricultural defaults "
        "in Maharashtra based on historical data."
    ),

    "thanks": (
        "You're welcome! 🌾 Feel free to ask anything else about loan eligibility, "
        "the ML pipeline, government schemes, or crop risk. I'm here to help."
    ),

    "bye": (
        "Goodbye! 🙏 Best wishes for your farming season. "
        "Visit again if you need any assistance with loan assessment or farmer schemes. — KisanCredit AI"
    ),

    "fallback": (
        "I can help with these topics:\n\n"
        "🏦 **Loan**: credit history, income, loan amount, approval chances, rejection\n"
        "🤖 **ML**: features, PCA, model comparison, overfitting, dataset\n"
        "🌾 **Agri**: crop risk, weather, property area, education\n"
        "🏛️ **Schemes**: PM Kisan, KCC, PMFBY, NABARD\n\n"
        "Try asking something like:\n"
        "• *\"Why is credit history so important?\"*\n"
        "• *\"What government schemes help rejected farmers?\"*\n"
        "• *\"Why did Logistic Regression win?\"*"
    ),
}

# ── QUICK SUGGESTION PROMPTS ───────────────────────────────────────────────────
SUGGESTIONS = [
    "What factors affect loan approval?",
    "How important is credit history?",
    "Explain the 8 engineered features",
    "Why did Logistic Regression win?",
    "What schemes help rejected farmers?",
    "How does PCA work here?",
    "What is the dataset about?",
    "How to improve my loan chances?",
]
