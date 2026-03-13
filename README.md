# 🧠 Geriatric Polypharmacy Clinical Decision Support System (CDSS)

**An AI-powered web application for real-time polypharmacy risk assessment in geriatric patients — combining Machine Learning, Explainable AI (SHAP), Retrieval-Augmented Generation (RAG), and Large Language Models.**

[Features](#-features) • [Tech Stack](#-tech-stack) • [Architecture](#-system-architecture) • [Setup](#-installation--setup) • [Usage](#-how-to-use) • [Dataset](#-dataset)

</div>

---

## 🎯 Problem Statement

Polypharmacy — the use of 5 or more medications simultaneously — affects over **40% of elderly patients (65+)** and is a leading cause of preventable hospital admissions. Clinicians lack real-time, explainable tools to assess the combined risk of multiple drug regimens in older adults with complex comorbidities.

This system bridges that gap.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **ADE Risk Prediction** | ML model predicts Adverse Drug Event probability with AUC 0.90 |
| 💊 **Treatment Effectiveness Score** | Random Forest regressor scores medication regimen effectiveness |
| 🔍 **SHAP Explainability** | Per-patient feature contribution analysis — not just predictions, but *why* |
| 📚 **RAG Clinical Evidence** | Retrieves real-time evidence from AGS Beers Criteria, ACC/AHA, FDA guidelines |
| 🧬 **Gemini 2.5 Flash LLM** | Conversational AI for clinical queries, image analysis, and report generation |
| 📊 **Medication Timeline** | Tracks patient risk scores over time for longitudinal monitoring |
| 🖼️ **Multimodal Input** | Upload prescription images or lab reports for AI analysis |
| 🔐 **Secure Auth** | User registration/login with hashed passwords and session management |
| 📋 **HTML Clinical Reports** | Auto-generated professional reports with SHAP findings and RAG evidence |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INPUT                              │
│         (Patient Age, Medications, Creatinine, etc.)        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  FLASK WEB APPLICATION                       │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │   Gemini    │  │  ML Models   │  │    RAG System     │  │
│  │  2.5 Flash  │  │              │  │                   │  │
│  │             │  │ • Logistic   │  │ • ChromaDB        │  │
│  │ • Feature   │  │   Regression │  │ • all-MiniLM-L6   │  │
│  │   Extraction│  │   (ADE Risk) │  │ • Medical KB      │  │
│  │ • Chat      │  │ • Random     │  │   (Beers, DDI,    │  │
│  │ • Reports   │  │   Forest     │  │    FDA, ACC/AHA)  │  │
│  │             │  │   (Efficacy) │  │                   │  │
│  └─────────────┘  └──────┬───────┘  └────────┬──────────┘  │
│                          │                   │             │
│                   ┌──────▼───────┐           │             │
│                   │    SHAP      │           │             │
│                   │ Explainer    │           │             │
│                   │ (XAI Layer)  │           │             │
│                   └──────┬───────┘           │             │
│                          │                   │             │
└──────────────────────────┼───────────────────┼─────────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────────────────────┐
                    │   HTML CLINICAL REPORT       │
                    │  • Risk Scores               │
                    │  • SHAP Feature Importance   │
                    │  • RAG Evidence + Sources    │
                    │  • Recommendations           │
                    └─────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend
- **Python 3.11** — Core language
- **Flask** — Web framework
- **Waitress** — Production WSGI server (Windows-compatible)
- **SQLAlchemy + SQLite** — User data & chat history persistence

### Machine Learning
- **Scikit-learn** — Logistic Regression (ADE classifier) + Random Forest (effectiveness regressor)
- **SHAP** — Explainable AI for per-patient feature contribution analysis
- **Pandas / NumPy** — Feature engineering pipeline
- **Joblib** — Model serialization

### AI & NLP
- **Google Gemini 2.5 Flash** — LLM for feature extraction, chat, and report generation
- **ChromaDB** — Vector database for RAG knowledge retrieval
- **Sentence Transformers (all-MiniLM-L6-v2)** — Medical text embeddings

### Frontend
- **HTML/CSS/JavaScript** — Responsive clinical UI
- **Jinja2** — Server-side templating

### Data
- **FAERS Dataset** — FDA Adverse Event Reporting System (5,087 geriatric records)

---

## 📊 Model Performance

| Model | Metric | Score |
|---|---|---|
| ADE Risk Classifier (Logistic Regression) | AUC | **0.90** |
| ADE Risk Classifier | CV AUC (5-Fold) | **0.904 ± 0.019** |
| ADE Risk Classifier | Accuracy | **82.3%** |
| Treatment Effectiveness Regressor (Random Forest) | R² Score | **0.737** |
| Treatment Effectiveness Regressor | RMSE | **0.051** |

### Key Risk Features (SHAP-ranked)
1. **Polypharmacy Count** — Number of concurrent medications
2. **DDI Count** — Drug-Drug Interaction count
3. **Age** — Patient age (65–99)
4. **PIM Count** — Potentially Inappropriate Medications (Beers Criteria)
5. **eGFR Category** — Renal function classification
6. **Comorbidity Index** — Chronic disease burden

---

## 🗂️ Project Structure

```
POLYPHARMACY CHATBOT/
│
├── app.py                    # Main Flask application
├── rag_system.py             # RAG system with ChromaDB + medical knowledge base
├── model_trainer.py          # ML model training pipeline
│
├── templates/                # HTML templates
│   ├── login.html
│   ├── register.html
│   ├── index.html            # Main chat interface
│   └── settings.html
│
├── static/                   # CSS, JS, assets
│
├── medical_knowledge_db/     # ChromaDB vector store (auto-generated)
│
├── FAERS dataset.csv         # Training data
├── ade classifier.pkl        # Trained ADE risk model
├── effectiveness regressor.pkl
├── shap_background data.pkl
├── feature_names.pkl
│
├── database.db               # SQLite user database (auto-generated)
├── .env                      # API keys (not committed)
└── requirements.txt
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.11+
- Google Gemini API key ([Get one free](https://makersuite.google.com/app/apikey))

### 1. Clone the repository
```bash
git clone https://github.com/jananidelli14/Geriatric-Polypharmacy-CDSS.git
cd Geriatric-Polypharmacy-CDSS
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
FLASK_SECRET_KEY=your_secret_key_here
```

### 4. Train the ML models
```bash
python model_trainer.py
```
This generates `ade classifier.pkl`, `effectiveness regressor.pkl`, `shap_background data.pkl`, and `feature_names.pkl`.

### 5. Start the application
```bash
waitress-serve --host=127.0.0.1 --port=5000 app:app
```

### 6. Open in browser
```
http://127.0.0.1:5000
```

---

## 🚀 How to Use

### Clinical Risk Analysis
Enter patient data in natural language:
```
Patient is 78 years old, female. Creatinine: 1.8 mg/dL.
Medications: Warfarin 5mg daily, Metoprolol 50mg BID,
Omeprazole 20mg daily, Gabapentin 300mg TID,
Amitriptyline 25mg nightly, Furosemide 40mg daily.
```

The system will generate a full clinical report with:
- ADE risk score with HIGH/MODERATE/LOW classification
- Treatment effectiveness score
- SHAP-based explanation of key risk drivers
- RAG-retrieved evidence from clinical guidelines
- Prioritized recommendations

### Conversational Mode
Ask general clinical questions:
```
What are the Beers Criteria medications to avoid in elderly patients?
What are the signs of digoxin toxicity?
```

### Image Analysis
Upload prescription images or lab reports for AI-powered analysis.

---

## 📋 Requirements

```
flask
flask-sqlalchemy
werkzeug
waitress
python-dotenv
google-generativeai
scikit-learn
shap
pandas
numpy
joblib
chromadb
sentence-transformers
Pillow
```

---

## 🔒 Medical Disclaimer

> This system is a **clinical decision support tool** intended for use by qualified healthcare professionals. It does not replace clinical judgment. All medical decisions must be made by licensed clinicians considering the complete patient context. This tool has not been approved by the FDA or any regulatory body for clinical use.

---

## 👩‍💻 Author

**Janani Delli**
- GitHub: [@jananidelli14](https://github.com/jananidelli14)

---

## 📄 License

This project is for educational and research purposes.

---


