"""
chatbot/knowledge_base.py
Enhanced knowledge base for KisanCredit AI chatbot.
Covers farmer schemes, loan eligibility, crop info, weather, and ML pipeline.
Supports English, Hindi, and Marathi keyword matching.
"""

# ── INTENT KEYWORDS (English + Hindi + Marathi) ────────────────────────────────
INTENTS = {
    "greeting": [
        "hi", "hello", "hey", "namaste", "namaskar", "good morning",
        "good afternoon", "good evening", "start", "help",
        # Hindi
        "नमस्ते", "हेलो", "शुरू", "मदद",
        # Marathi
        "नमस्कार", "सुरुवात", "मदत",
    ],

    "credit_history": [
        "credit history", "credit score", "cibil", "repayment",
        "past loans", "credit record", "credit_history",
        # Hindi
        "क्रेडिट", "सिबिल", "उधार", "चुकौती", "पुराना लोन",
        # Marathi
        "कर्ज इतिहास", "परतफेड", "क्रेडिट",
    ],

    "income": [
        "income", "salary", "earnings", "co-applicant", "coapplicant",
        "total income", "combined income", "monthly income",
        # Hindi
        "आय", "वेतन", "कमाई", "मासिक आय", "सह-आवेदक",
        # Marathi
        "उत्पन्न", "पगार", "मासिक उत्पन्न", "एकत्रित उत्पन्न",
    ],

    "loan_amount": [
        "loan amount", "how much", "borrow", "amount", "loanamount",
        "how much loan", "maximum loan",
        # Hindi
        "ऋण राशि", "लोन राशि", "कितना", "उधार", "अधिकतम लोन",
        # Marathi
        "कर्ज रक्कम", "किती", "जास्तीत जास्त कर्ज",
    ],

    "features": [
        "feature", "engineered", "8 features", "variables", "columns",
        "emi", "balance income", "debt to income", "components",
        # Hindi
        "विशेषता", "ईएमआई", "ऋण आय अनुपात",
        # Marathi
        "वैशिष्ट्ये", "ईएमआय",
    ],

    "pca": [
        "pca", "principal component", "dimensionality", "reduce features",
        "variance", "12 components", "22 features",
    ],

    "model": [
        "model", "algorithm", "logistic", "random forest",
        "gradient boosting", "which model", "best model", "accuracy",
        "auc", "roc", "pr-auc", "results", "performance",
        # Hindi
        "मॉडल", "एल्गोरिदम", "सटीकता",
        # Marathi
        "मॉडेल", "अचूकता",
    ],

    "approval": [
        "approve", "approval", "chance", "probability",
        "will i get", "eligible", "qualify", "eligibility",
        # Hindi
        "स्वीकृति", "पात्रता", "मंजूरी", "योग्यता",
        # Marathi
        "मंजुरी", "पात्रता", "अर्ज मंजूर",
    ],

    "rejection": [
        "reject", "rejected", "denied", "decline", "not approved",
        "loan rejected", "alternative", "what if rejected",
        # Hindi
        "अस्वीकृत", "रद्द", "नामंजूर", "क्या करें",
        # Marathi
        "नाकारले", "रद्द", "नामंजूर",
    ],

    "schemes": [
        "scheme", "government", "pm kisan", "kcc", "kisan credit",
        "nabard", "pmfby", "subsidy", "crop insurance", "support",
        "fasal bima", "soil health", "agri", "kisaan", "sarkar",
        "yojana", "benefit", "pension", "mgnrega", "mandhan",
        # Hindi
        "योजना", "सरकारी", "किसान सम्मान", "किसान क्रेडिट कार्ड",
        "फसल बीमा", "अनुदान", "सब्सिडी", "पेंशन", "मनरेगा",
        "किसान", "सरकार",
        # Marathi
        "योजना", "शेतकरी", "सरकारी", "पीक विमा", "अनुदान",
        "सबसिडी", "निवृत्तीवेतन",
    ],

    "kcc": [
        "kcc", "kisan credit card", "किसान क्रेडिट कार्ड", "kcc card",
        "revolving credit", "4 percent", "3 lakh", "kisan card",
    ],

    "pmfby": [
        "pmfby", "fasal bima", "crop insurance", "फसल बीमा", "पीक विमा",
        "pradhan mantri fasal", "insurance premium", "natural disaster",
    ],

    "pm_kisan": [
        "pm kisan", "pm-kisan", "samman nidhi", "6000", "direct transfer",
        "किसान सम्मान निधि", "पीएम किसान", "installment", "annual benefit",
    ],

    "nabard": [
        "nabard", "ridf", "rural infrastructure", "cooperative",
        "शहरी", "ग्रामीण", "nabard loan", "rural development",
    ],

    "dataset": [
        "dataset", "data", "kaggle", "614", "rows", "training data",
        "loan_train", "real applicants",
    ],

    "overfitting": [
        "overfit", "overfitting", "generalise", "generalize", "cv gap",
        "cross validation", "validation",
    ],

    "property": [
        "property", "area", "rural", "urban", "semiurban", "location", "region",
        # Hindi
        "संपत्ति", "ग्रामीण", "शहरी", "क्षेत्र",
        # Marathi
        "मालमत्ता", "ग्रामीण", "शहरी",
    ],

    "education": [
        "education", "graduate", "not graduate", "qualification", "degree",
        # Hindi
        "शिक्षा", "स्नातक", "डिग्री",
        # Marathi
        "शिक्षण", "पदवी",
    ],

    "crop": [
        "crop", "wheat", "rice", "cotton", "soybean", "sugarcane",
        "vegetable", "kharif", "rabi", "farming", "kheti", "fasal",
        # Hindi
        "फसल", "गेहूं", "चावल", "कपास", "सोयाबीन", "गन्ना",
        "खेती", "किसानी",
        # Marathi
        "पीक", "गहू", "तांदूळ", "कापूस", "सोयाबीन", "ऊस",
        "शेती",
    ],

    "weather": [
        "weather", "flood", "drought", "rain", "temperature", "heat",
        "monsoon", "climate", "risk",
        # Hindi
        "मौसम", "बाढ़", "सूखा", "बारिश", "गर्मी", "मानसून",
        # Marathi
        "हवामान", "पूर", "दुष्काळ", "पाऊस", "मान्सून",
    ],

    "soil": [
        "soil", "soil health", "soil card", "mitti", "fertilizer",
        "nutrient", "soil testing",
        # Hindi
        "मिट्टी", "मृदा", "उर्वरक", "मृदा स्वास्थ्य कार्ड",
        # Marathi
        "माती", "मृदा", "खते", "माती आरोग्य कार्ड",
    ],

    "interest_rate": [
        "interest", "interest rate", "rate", "percent", "4%", "7%",
        "byaj", "ब्याज",
        # Hindi
        "ब्याज दर", "प्रतिशत",
        # Marathi
        "व्याज दर", "टक्के",
    ],

    "documents": [
        "document", "papers", "required", "aadhaar", "aadhar", "pan",
        "land record", "7/12", "satbara", "passbook",
        # Hindi
        "दस्तावेज़", "कागजात", "आधार", "पैन", "भूमि रिकॉर्ड",
        # Marathi
        "कागदपत्रे", "आधार", "पॅन", "सातबारा", "जमीन नोंद",
    ],

    "thanks": [
        "thank", "thanks", "thank you", "shukriya", "dhanyavaad",
        "great", "awesome", "nice", "good",
        # Hindi
        "धन्यवाद", "शुक्रिया", "बहुत अच्छा",
        # Marathi
        "धन्यवाद", "आभार", "छान",
    ],

    "bye": [
        "bye", "goodbye", "see you", "exit", "quit", "done",
        # Hindi
        "अलविदा", "बाय", "फिर मिलेंगे",
        # Marathi
        "निरोप", "पुन्हा भेटू",
    ],

    "language": [
        "hindi", "marathi", "english", "language", "bhasha",
        "हिंदी", "मराठी", "भाषा",
    ],
}

# ── RESPONSES ──────────────────────────────────────────────────────────────────
RESPONSES = {
    "greeting": (
        "🙏 Namaste! I'm **KisanCredit AI Assistant**.\n\n"
        "I can help you with:\n"
        "• 🏦 **Loan eligibility** — approval factors, credit score, income\n"
        "• 🏛️ **Government schemes** — PM Kisan, KCC, PMFBY, NABARD and more\n"
        "• 🌾 **Crop & weather risk** — seasonal advice, climate impact\n"
        "• 📋 **Documents required** — what to prepare for loan application\n"
        "• 🤖 **ML pipeline** — how the prediction model works\n\n"
        "💬 *You can ask in English, Hindi (हिंदी), or Marathi (मराठी)!*\n\n"
        "What would you like to know? 🌾"
    ),

    "language": (
        "🌐 **Supported Languages / भाषाएं / भाषा:**\n\n"
        "• 🇬🇧 **English** — Full support\n"
        "• 🇮🇳 **Hindi (हिंदी)** — पूरा समर्थन\n"
        "• 🟠 **Marathi (मराठी)** — पूर्ण समर्थन\n\n"
        "You can type or speak your question in any of these languages.\n"
        "हिंदी में पूछें: *\"किसान क्रेडिट कार्ड के बारे में बताएं\"*\n"
        "मराठीत विचारा: *\"पीक विम्याबद्दल सांगा\"*"
    ),

    "credit_history": (
        "**Credit History** is the single most important factor — **38% weight** in the model.\n\n"
        "📊 Stats from Kaggle dataset (614 applicants):\n"
        "• Credit_History = 1 (clean record): **87% approval rate**\n"
        "• Credit_History = 0 (defaults/dues): only **14% approval rate**\n\n"
        "💡 **How to improve your credit history:**\n"
        "• Repay existing KCC / crop loans on time\n"
        "• Clear any pending overdues before applying\n"
        "• Maintain 6–12 months clean repayment record\n"
        "• Get a free CIBIL report at ciscore.in\n\n"
        "🏦 *Check credit history at your nearest CSC (Common Service Centre)*\n\n"
        "**हिंदी:** क्रेडिट इतिहास सबसे महत्वपूर्ण है — 38% वजन। साफ रिकॉर्ड से 87% मंजूरी।"
    ),

    "income": (
        "**Income** is the 2nd most important factor (~**20% weight**).\n\n"
        "Model uses **TotalIncome = ApplicantIncome + CoapplicantIncome**.\n\n"
        "📊 Benchmarks:\n"
        "• TotalIncome > ₹6,000/month → **81% approval rate**\n"
        "• TotalIncome < ₹2,500/month → high risk, likely rejection\n\n"
        "💡 **Tips to improve:**\n"
        "• Add spouse/family as co-applicant to boost combined income\n"
        "• Include all income sources: farming + dairy + other\n"
        "• Get income certificate from Tehsildar office\n\n"
        "**हिंदी:** आय सबसे महत्वपूर्ण कारकों में से एक है। ₹6,000/माह से अधिक संयुक्त आय से 81% मंजूरी।\n"
        "**मराठी:** उत्पन्न महत्त्वाचे आहे. एकत्रित उत्पन्न ₹6,000/महिना असल्यास 81% मंजुरी."
    ),

    "loan_amount": (
        "**Loan Amount** carries ~**13% weight** in the model.\n\n"
        "📊 Dataset benchmarks:\n"
        "• Median loan: ₹1,28,000 — conservative amounts improve approval\n"
        "• Loans > ₹4,00,000 are flagged as high-exposure\n\n"
        "💡 **Formula to use:**\n"
        "• Keep EMI ≤ 35% of monthly income for best approval\n"
        "• Debt-to-Income ratio < 0.4 is ideal\n"
        "• Under KCC: up to ₹3 lakh at just 4% interest\n\n"
        "**हिंदी:** ऋण राशि को आय के अनुपात में रखें। DTI < 0.4 सबसे अच्छा है।"
    ),

    "approval": (
        "**Key Factors for Loan Approval** (ranked by importance):\n\n"
        "1. 🏆 **Credit History** (38%) — must be clean; most critical\n"
        "2. 💰 **Total Income** (20%) — aim for > ₹6,000/month combined\n"
        "3. 📋 **Loan Amount** (13%) — keep Debt-to-Income ratio < 0.4\n"
        "4. 📉 **Debt-to-Income** (11%) — lower is better\n"
        "5. 🧾 **EMI burden** (7%) — should be manageable vs income\n"
        "6. 🎓 **Education** (8%) — Graduate has lower default rate\n"
        "7. 🏘️ **Property Area** (5%) — Semiurban has best approval (76.3%)\n\n"
        "✅ **Quick checklist:**\n"
        "• Clean credit record for 12+ months\n"
        "• Combined income > ₹4,000/month\n"
        "• Loan ≤ 40% of annual income\n"
        "• Land documents ready\n\n"
        "**हिंदी:** ऋण मंजूरी के लिए: साफ क्रेडिट इतिहास, अच्छी आय, उचित राशि।"
    ),

    "rejection": (
        "😟 If your loan is **rejected**, don't worry — here are your options:\n\n"
        "🏛️ **Government Alternatives:**\n"
        "• **PM Kisan Samman Nidhi** — ₹6,000/year direct to bank (no loan needed)\n"
        "• **Kisan Credit Card (KCC)** — up to ₹3 lakh @ 4% interest\n"
        "• **PMFBY** — Crop insurance protects against loss\n"
        "• **SHG Micro-credit** — Self-Help Group loans at low rates\n"
        "• **NABARD** — Rural cooperative bank loans\n"
        "• **PM SVANidhi** — Micro loans for rural self-employment\n\n"
        "📈 **Improve Your Profile (reapply in 6 months):**\n"
        "• Clear all existing defaults (biggest factor — 38%)\n"
        "• Add a co-applicant (spouse/parent) to boost income\n"
        "• Reduce loan amount or extend tenure\n"
        "• Provide land/property as collateral\n\n"
        "📞 **Helplines:** PM Kisan: 155261 | PMFBY: 14447 | NABARD: 1800-22-0000\n\n"
        "**हिंदी:** अस्वीकृति पर: KCC, PM किसान, PMFBY के लिए आवेदन करें। 6 महीने में सुधार करें और फिर आवेदन करें।"
    ),

    "schemes": (
        "**🏛️ Government Schemes for Indian Farmers:**\n\n"
        "| Scheme | Benefit | Who Can Apply |\n"
        "|---|---|---|\n"
        "| **PM Kisan Samman Nidhi** | ₹6,000/year (3 × ₹2,000) | All landholding farmers |\n"
        "| **Kisan Credit Card (KCC)** | Credit up to ₹3L @ 4% p.a. | Any farmer with land record |\n"
        "| **PMFBY** | Crop insurance, subsidised premium | All notified crops |\n"
        "| **PM Kisan MAN DHAN** | Pension ₹3,000/month at age 60 | Small & marginal farmers |\n"
        "| **Soil Health Card** | Free soil test + fertilizer advice | All farmers |\n"
        "| **NABARD RIDF** | Infrastructure loans | Via state cooperatives |\n"
        "| **E-NAM** | Online crop price + trading | Registered mandis |\n"
        "| **MGNREGA** | 100 days guaranteed rural work | Rural households |\n"
        "| **PM SVANidhi** | ₹10K–₹50K micro loans | Small rural vendors |\n\n"
        "📞 **Helplines:** PM Kisan: **155261** | PMFBY: **14447** | NABARD: **1800-22-0000**\n\n"
        "**हिंदी:** सरकारी योजनाएं: PM किसान (₹6000/साल), KCC (₹3 लाख 4% पर), PMFBY (फसल बीमा)।\n"
        "**मराठी:** सरकारी योजना: PM किसान (₹6000/वर्ष), KCC (₹3 लाख 4% वर), PMFBY (पीक विमा)."
    ),

    "kcc": (
        "💳 **Kisan Credit Card (KCC) — Complete Guide:**\n\n"
        "🎯 **What is it?** A revolving credit card for farmers with flexible repayment.\n\n"
        "💰 **Benefits:**\n"
        "• Credit limit: up to **₹3 lakh** (₹1.6L without collateral)\n"
        "• Interest rate: **4% p.a.** (after 2% government interest subvention)\n"
        "• Valid for **5 years**, renewed annually\n"
        "• No processing fee for loans up to ₹3 lakh\n"
        "• Personal accident insurance of ₹50,000 included\n\n"
        "📋 **Eligibility:**\n"
        "• All farmers — individual, tenant, sharecropper, SHG\n"
        "• Must have land records (7/12 or equivalent)\n"
        "• No minimum income requirement\n\n"
        "📝 **Documents needed:**\n"
        "• Aadhaar card, PAN card\n"
        "• Land records / 7-12 extract\n"
        "• Passport-size photos\n"
        "• Bank account passbook\n\n"
        "🏦 Apply at: Any nationalised bank, NABARD cooperative, or online at **pmkisan.gov.in**\n\n"
        "**हिंदी:** KCC से ₹3 लाख तक 4% ब्याज पर। बिना गारंटी ₹1.6 लाख तक। आधार, पैन, भूमि दस्तावेज चाहिए।"
    ),

    "pmfby": (
        "🌾 **PMFBY — Pradhan Mantri Fasal Bima Yojana (Crop Insurance):**\n\n"
        "🎯 **What does it cover?**\n"
        "• Natural calamities: flood, drought, hailstorm, landslide\n"
        "• Pest & disease attacks on standing crop\n"
        "• Post-harvest losses for 2 weeks after harvest\n\n"
        "💰 **Premium Rates (farmer's share only):**\n"
        "| Crop Season | Max Premium |\n"
        "|---|---|\n"
        "| Kharif crops | **2%** of sum insured |\n"
        "| Rabi crops | **1.5%** of sum insured |\n"
        "| Commercial/Horticultural | **5%** of sum insured |\n\n"
        "📋 **Who can apply?** All farmers growing notified crops.\n"
        "• Mandatory for KCC loan holders\n"
        "• Optional / voluntary for others\n\n"
        "📝 **How to apply:** Through your bank, CSC (Common Service Centre), or\n"
        "🌐 **pmfby.gov.in** | 📞 Helpline: **14447**\n\n"
        "**हिंदी:** PMFBY: खरीफ में 2%, रबी में 1.5% प्रीमियम। बाढ़, सूखा, ओलावृष्टि से सुरक्षा।\n"
        "**मराठी:** PMFBY: खरीप 2%, रब्बी 1.5% हप्ता. पूर, दुष्काळ, गारपीटपासून संरक्षण."
    ),

    "pm_kisan": (
        "🌱 **PM Kisan Samman Nidhi — Complete Guide:**\n\n"
        "💰 **Benefit:** ₹6,000 per year in **3 instalments of ₹2,000**\n"
        "• Directly deposited to bank account (DBT)\n"
        "• No middleman — goes straight to your account!\n\n"
        "✅ **Who is eligible?**\n"
        "• All farmer families with cultivable land\n"
        "• Small and marginal farmers especially\n"
        "• Both agricultural and non-agricultural land owners\n\n"
        "❌ **Who is NOT eligible?**\n"
        "• Government employees, income tax payers\n"
        "• Institutional landholders\n"
        "• Pensioners receiving ≥ ₹10,000/month\n\n"
        "📝 **How to register?**\n"
        "1. Visit **pmkisan.gov.in**\n"
        "2. Click 'New Farmer Registration'\n"
        "3. Enter Aadhaar + land details\n"
        "4. Or visit nearest CSC / bank\n\n"
        "📞 Helpline: **155261** (free, 24×7)\n\n"
        "**हिंदी:** PM किसान: ₹6000/साल, 3 किश्तों में। pmkisan.gov.in पर पंजीकरण करें।\n"
        "**मराठी:** PM किसान: ₹6000/वर्ष, 3 हप्त्यांमध्ये. pmkisan.gov.in वर नोंदणी करा."
    ),

    "nabard": (
        "🏦 **NABARD — National Bank for Agriculture and Rural Development:**\n\n"
        "🎯 **Key schemes through NABARD:**\n\n"
        "• **RIDF (Rural Infrastructure Development Fund)**\n"
        "  — Loans to state govts for irrigation, bridges, roads\n\n"
        "• **Kisan Credit Card** via cooperative banks\n"
        "  — Farmers can apply at NABARD-affiliated cooperatives\n\n"
        "• **NABARD Refinance** for microfinance institutions\n"
        "  — SHG-Bank Linkage Programme for landless farmers\n\n"
        "• **Wadi Programme** — Horticulture + tribal farmer support\n\n"
        "• **Farmer Producer Organisations (FPO) support**\n"
        "  — Collective strength for better price + credit\n\n"
        "📞 NABARD Helpline: **1800-22-0000** (toll free)\n"
        "🌐 Website: **nabard.org**\n\n"
        "**हिंदी:** NABARD ग्रामीण बैंकों और सहकारी संस्थाओं के माध्यम से किसानों को ऋण देता है।"
    ),

    "soil": (
        "🌱 **Soil Health Card Scheme:**\n\n"
        "• Free soil testing every **2 years** for all farmers\n"
        "• Card shows 12 parameters: NPK, pH, micro-nutrients\n"
        "• Fertilizer recommendations specific to your field\n"
        "• Can improve yield by **10–15%** by optimising input use\n"
        "• Reduces fertilizer over-spending by up to 20%\n\n"
        "📍 **How to get it:**\n"
        "1. Visit your local Agriculture Department office or Krishi Vigyan Kendra (KVK)\n"
        "2. Submit soil sample from your field\n"
        "3. Receive card within 30 days\n"
        "4. Online: **soilhealth.dac.gov.in**\n\n"
        "**हिंदी:** मृदा स्वास्थ्य कार्ड: मुफ्त मिट्टी जांच। उर्वरक सलाह से 10-15% अधिक उपज।"
    ),

    "interest_rate": (
        "💹 **Interest Rates for Farmer Loans:**\n\n"
        "| Loan Type | Interest Rate | Notes |\n"
        "|---|---|---|\n"
        "| **KCC (Kisan Credit Card)** | **4% p.a.** | After govt subvention |\n"
        "| KCC > ₹3 lakh | 7% p.a. | Standard rate |\n"
        "| Agricultural term loans | 7–9% | Varies by bank |\n"
        "| Micro-finance (SHG) | 10–14% | Via NABARD linkage |\n"
        "| Cooperative bank loans | 5–7% | State-specific |\n"
        "| NBFC / private lender | 18–24% | Avoid if possible |\n\n"
        "💡 **Always prefer:** Nationalised bank → cooperative → NBFC\n"
        "📞 For subsidised rates, contact: your nearest **Krishi Shakha** (agriculture branch)\n\n"
        "**हिंदी:** KCC पर 4% ब्याज (सरकारी सब्सिडी के बाद)। राष्ट्रीय बैंक सबसे सस्ता।"
    ),

    "documents": (
        "📋 **Documents Required for Farmer Loan Application:**\n\n"
        "✅ **Identity Proof** (any one):\n"
        "• Aadhaar Card (mandatory)\n"
        "• PAN Card\n"
        "• Voter ID\n\n"
        "✅ **Land Records** (all required):\n"
        "• 7/12 Extract (Satbara Utara in Marathi / खसरा-खतौनी in Hindi)\n"
        "• Mutation register (Ferfar / दाखिल-खारिज)\n"
        "• Land ownership certificate from Tehsildar\n\n"
        "✅ **Income & Bank:**\n"
        "• Last 6 months bank statements\n"
        "• Income certificate from revenue officer\n"
        "• Passbook with active account\n\n"
        "✅ **Photographs:**\n"
        "• 2–4 recent passport-size photos\n\n"
        "💡 Keep **photocopies** of all documents. Many banks accept self-attested copies.\n\n"
        "**मराठी:** कर्जासाठी: आधार, पॅन, सातबारा उतारा, बँक पासबुक, उत्पन्नाचा दाखला.\n"
        "**हिंदी:** लोन के लिए: आधार, पैन, खसरा-खतौनी, बैंक पासबुक, आय प्रमाण पत्र।"
    ),

    "features": (
        "**8 Engineered Features** added to original 14 Kaggle columns:\n\n"
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
        "Plus **3 bin features** (KBinsDiscretizer, quantile, n=5) → **22 total features** before PCA."
    ),

    "pca": (
        "**PCA (Principal Component Analysis):**\n\n"
        "🔢 22 features → **12 principal components** (retaining **95% variance**)\n\n"
        "Why PCA helps:\n"
        "• Eliminates **multicollinearity** between TotalIncome / EMI / LoanToIncome\n"
        "• Acts as **regularisation** — reduces overfitting risk\n"
        "• Speeds up training by ~25%\n"
        "• LogisticRegression performs best on decorrelated features"
    ),

    "model": (
        "**Real Kaggle Results** (491 train / 123 test, Stratified 80/20):\n\n"
        "| Model | Test Accuracy | ROC-AUC | PR-AUC |\n"
        "|---|---|---|---|\n"
        "| ✅ **Logistic Regression** | **80.49%** | **0.8796** | **0.9439** |\n"
        "| Random Forest | 76.42% | 0.8424 | 0.9211 |\n"
        "| Gradient Boosting | 76.42% | 0.8015 | 0.8847 |\n\n"
        "**Why Logistic Regression won:** PCA-decorrelated features play to LR's strengths. "
        "CV-Test AUC gap = 0.022 — no overfitting."
    ),

    "dataset": (
        "**Kaggle Loan Prediction Dataset:**\n\n"
        "• **614 rows** × 14 columns\n"
        "• Approved (Y): 422 (68.7%) | Rejected (N): 192 (31.3%)\n"
        "• Missing values: Credit_History (54), Self_Employed (23), LoanAmount (22)\n"
        "• Source: kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset"
    ),

    "overfitting": (
        "**Overfitting Prevention:**\n\n"
        "• 5-Fold Stratified CV — CV-Test AUC gap = 0.022 ✅\n"
        "• PCA regularisation — 22→12 components removes noise\n"
        "• L2 regularisation in LogisticRegression (C=0.8)\n"
        "• RF max_depth=8 + GB subsample=0.8"
    ),

    "property": (
        "**Property Area** and approval rates:\n\n"
        "| Area | Approval Rate |\n"
        "|---|---|\n"
        "| Semiurban | **76.3%** (best) |\n"
        "| Urban | 67.8% |\n"
        "| Rural | 61.2% |\n\n"
        "Semiurban applicants benefit from better infrastructure and more stable incomes."
    ),

    "education": (
        "**Education** carries ~8% weight in the model.\n\n"
        "• **Graduates**: Lower default rate → positive approval signal (71% approval)\n"
        "• **Non-graduates**: Not disqualifying alone, but combined with other risk factors reduces chances (61% approval)\n\n"
        "📚 Many schemes like KCC do NOT require education qualification."
    ),

    "crop": (
        "**Crop Risk Profile (Kharif vs Rabi):**\n\n"
        "| Crop | Kharif Risk | Rabi Risk | Notes |\n"
        "|---|---|---|---|\n"
        "| Cotton | **65%** | 22% | High kharif exposure |\n"
        "| Soybean | **55%** | 40% | Moisture-sensitive |\n"
        "| Rice/Paddy | 42% | 20% | Flood risk |\n"
        "| Vegetables | 38% | 28% | Short cycle volatility |\n"
        "| Sugarcane | 30% | 45% | Assured MSP procurement |\n"
        "| Wheat | 18% | **35%** | Stable, low kharif risk |\n\n"
        "Sugarcane and wheat are **lower-risk** due to MSP protection.\n"
        "💡 Get PMFBY insurance for any high-risk crop!\n\n"
        "**हिंदी:** कपास और सोयाबीन खरीफ में उच्च जोखिम। गेहूं और गन्ना अपेक्षाकृत सुरक्षित।\n"
        "**मराठी:** कापूस आणि सोयाबीन खरीप हंगामात जास्त धोका. गहू आणि ऊस तुलनेने सुरक्षित."
    ),

    "weather": (
        "**Weather & Climate Risk in Loan Assessment:**\n\n"
        "🌵 **Drought**: Raises default risk by 18–25%. Cotton & soybean most vulnerable.\n"
        "🌊 **Flood**: Flood-prone districts attract +15% risk premium per RBI norms.\n"
        "🌡️ **Heat Stress**: Temperatures > 40°C can cut crop yield by 30–40%.\n"
        "🌧️ **Monsoon Variability**: El Niño years → ~22% higher defaults in Maharashtra.\n\n"
        "🛡️ **Protection:** Apply for **PMFBY** crop insurance to cover weather losses!\n"
        "📞 PMFBY Helpline: **14447**\n\n"
        "**हिंदी:** सूखा, बाढ़, गर्मी से ऋण जोखिम बढ़ता है। PMFBY बीमा लें।\n"
        "**मराठी:** दुष्काळ, पूर, उष्णतेमुळे कर्ज धोका वाढतो. PMFBY विमा घ्या."
    ),

    "thanks": (
        "🙏 You're welcome! Feel free to ask anything else about:\n"
        "• Loan eligibility & factors\n"
        "• Government schemes (PM Kisan, KCC, PMFBY)\n"
        "• Crop & weather risk\n"
        "• Documents required\n\n"
        "धन्यवाद! | धन्यवाद! 🌾"
    ),

    "bye": (
        "👋 Goodbye! Best wishes for your farming season.\n"
        "अलविदा! आपकी फसल अच्छी हो। 🌾\n"
        "निरोप! तुमच्या शेतीसाठी शुभेच्छा. 🌾\n\n"
        "— KisanCredit AI"
    ),

    "fallback": (
        "I can help with these topics:\n\n"
        "🏦 **Loan:** credit history, income, loan amount, approval, rejection\n"
        "🏛️ **Schemes:** PM Kisan, KCC, PMFBY, NABARD, Soil Health Card\n"
        "🌾 **Agri:** crop risk, weather, property area, education\n"
        "📋 **Documents:** Aadhaar, land records, income certificate\n"
        "🤖 **ML Pipeline:** features, PCA, model comparison\n\n"
        "Try asking:\n"
        "• *\"What is the Kisan Credit Card?\"*\n"
        "• *\"किसान सम्मान निधि के लिए कैसे आवेदन करें?\"*\n"
        "• *\"पीक विम्यासाठी कोणते कागदपत्रे लागतात?\"*"
    ),
}

# ── QUICK SUGGESTION PROMPTS ───────────────────────────────────────────────────
SUGGESTIONS = [
    "What is the Kisan Credit Card (KCC)?",
    "How to apply for PM Kisan Samman Nidhi?",
    "What crop insurance schemes are available?",
    "What documents are needed for a farm loan?",
    "How important is credit history for loan approval?",
    "What is PMFBY crop insurance?",
    "Which crops have the lowest loan risk?",
    "What government schemes help rejected farmers?",
    "How to improve my loan approval chances?",
    "What interest rate will I get on KCC?",
]

# ── MULTILINGUAL QUICK REPLIES ─────────────────────────────────────────────────
MULTILINGUAL_SUGGESTIONS = {
    "hi": [
        "किसान क्रेडिट कार्ड क्या है?",
        "PM किसान के लिए कैसे आवेदन करें?",
        "फसल बीमा कैसे लें?",
        "लोन के लिए कौन से दस्तावेज चाहिए?",
        "क्रेडिट इतिहास क्यों जरूरी है?",
    ],
    "mr": [
        "किसान क्रेडिट कार्ड म्हणजे काय?",
        "PM किसान साठी अर्ज कसा करावा?",
        "पीक विमा कसा मिळेल?",
        "कर्जासाठी कोणती कागदपत्रे लागतात?",
        "पत इतिहास का महत्त्वाचा आहे?",
    ],
    "en": SUGGESTIONS,
}
