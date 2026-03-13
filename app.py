import os
import sys
import re
import json
import logging
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import shap

from flask import (
    Flask, request, jsonify, render_template,
    redirect, url_for, flash, session
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# MULTIMODAL
import base64
from PIL import Image
from io import BytesIO
import google.generativeai as genai

# RAG system
from rag_system import initialize_rag_system

# BASIC SETUP
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get(
    'FLASK_SECRET_KEY',
    hashlib.sha256(b'geriatric_cdss_2025').hexdigest()
)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        try:
            handler.stream = open(sys.stdout.fileno(), mode='w',
                                  encoding='utf-8', buffering=1)
        except Exception:
            pass

app.logger.setLevel(logging.INFO)

# DATABASE MODELS
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey('user.id'),
        nullable=False
    )
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    message_type = db.Column(db.String(20), default='chat')
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class MedicationTimeline(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey('user.id'),
        nullable=False
    )
    patient_age = db.Column(db.Integer, nullable=False)
    medication_count = db.Column(db.Integer, nullable=False)
    ade_risk = db.Column(db.Float, nullable=False)
    beers_count = db.Column(db.Integer, default=0)
    ddi_count = db.Column(db.Integer, default=0)
    medications_snapshot = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    try:
        db.create_all()
        app.logger.info("Database tables created successfully")
    except Exception as e:
        app.logger.error(f"Error creating database tables: {e}")

# GEMINI CONFIG
API_KEY = os.environ.get("GEMINI_API_KEY")
try:
    if not API_KEY or "YOUR_GEMINI_API_KEY_HERE" in API_KEY or "AIzaSy..." in API_KEY:
        raise ValueError("GEMINI_API_KEY not properly configured")
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    app.logger.info("Gemini Model configured successfully")
except Exception as e:
    app.logger.error(f"GEMINI CONFIG ERROR: {e}")
    gemini_model = None

# ML MODEL + SHAP CONFIG
ADE_MODEL = None
EFFECTIVENESS_MODEL = None
SHAP_BACKGROUND_DATA = None
ADE_EXPLAINER = None
MODEL_FEATURES = None

try:
    ADE_MODEL = joblib.load('ade classifier.pkl')
    EFFECTIVENESS_MODEL = joblib.load('effectiveness regressor.pkl')

    try:
        SHAP_BACKGROUND_DATA = joblib.load('shap_background_data.pkl')
    except FileNotFoundError:
        SHAP_BACKGROUND_DATA = joblib.load('shap_background data.pkl')

    MODEL_FEATURES = joblib.load('feature_names.pkl')

    if isinstance(SHAP_BACKGROUND_DATA, (list, np.ndarray)):
        num_features = len(MODEL_FEATURES)
        bg_array = np.array(SHAP_BACKGROUND_DATA)

        if bg_array.ndim == 1:
            bg_array = bg_array.reshape(1, -1)

        if bg_array.shape[1] > num_features:
            bg_array = bg_array[:, :num_features]

        SHAP_BACKGROUND_DATA = pd.DataFrame(
            bg_array[:100].astype(np.float64),
            columns=MODEL_FEATURES
        )
    elif isinstance(SHAP_BACKGROUND_DATA, pd.DataFrame):
        SHAP_BACKGROUND_DATA = SHAP_BACKGROUND_DATA[MODEL_FEATURES].head(100)

    for col in SHAP_BACKGROUND_DATA.columns:
        SHAP_BACKGROUND_DATA[col] = pd.to_numeric(
            SHAP_BACKGROUND_DATA[col],
            errors='coerce'
        ).fillna(0.0).astype(np.float64)

    SHAP_BACKGROUND_DATA = SHAP_BACKGROUND_DATA.select_dtypes(include=[np.number])
    app.logger.info(
        f"Background data: shape={SHAP_BACKGROUND_DATA.shape}, "
        f"dtypes={SHAP_BACKGROUND_DATA.dtypes.unique()}"
    )

    if not SHAP_BACKGROUND_DATA.empty and len(SHAP_BACKGROUND_DATA) > 0:
        ADE_EXPLAINER = shap.Explainer(
            ADE_MODEL.predict_proba,
            SHAP_BACKGROUND_DATA.values.astype(np.float64),
            feature_names=MODEL_FEATURES
        )
        app.logger.info("SHAP Explainer initialized successfully")
    else:
        app.logger.error("Background data is empty!")
        ADE_EXPLAINER = None

    app.logger.info("SUCCESS: All ML Models and SHAP Explainer loaded successfully")

except FileNotFoundError as e:
    app.logger.warning(f"ML files not found: {e}. Running in conversation-only mode.")
    ADE_EXPLAINER = None
    MODEL_FEATURES = [
        'age', 'PolypharmacyCount', 'Comorbidity Index',
        'eGFR Category', 'PIM Count', 'DDI Count', 'Gender_Male'
    ]
except Exception as e:
    app.logger.error(f"ERROR: Failed loading ML models: {e}")
    import traceback
    app.logger.error(traceback.format_exc())
    ADE_EXPLAINER = None
    MODEL_FEATURES = [
        'age', 'PolypharmacyCount', 'Comorbidity Index',
        'eGFR Category', 'PIM Count', 'DDI Count', 'Gender_Male'
    ]

# RAG SYSTEM - initialized later in __main__
RAG_SYSTEM = None


def initialize_rag_once():
    global RAG_SYSTEM
    try:
        app.logger.info("Initializing Medical RAG System...")
        RAG_SYSTEM = initialize_rag_system()
        app.logger.info("SUCCESS: RAG system initialized")
    except Exception as e:
        app.logger.error(f"Failed to initialize RAG system: {e}")
        RAG_SYSTEM = None


def get_rag_system():
    return RAG_SYSTEM


# CONSTANTS
ADE_RISK_DRUGS = [
    "Warfarin", "Digoxin", "Insulin", "Gabapentin", "Amiodarone",
    "Prednisone", "Omeprazole",
    "Benzodiazepine", "Tramadol", "Amitriptyline", "NSAID",
    "Donepezil", "Rivastigmine", "Galantamine",
    "Diphenhydramine", "Hydroxyzine"
]

# TIMELINE HELPERS
def save_medication_timeline(user_id, patient_age, med_count,
                             ade_risk, beers, ddi, meds_text):
    try:
        timeline_entry = MedicationTimeline(
            user_id=user_id,
            patient_age=patient_age,
            medication_count=med_count,
            ade_risk=ade_risk,
            beers_count=beers,
            ddi_count=ddi,
            medications_snapshot=meds_text[:500] if meds_text else ""
        )
        db.session.add(timeline_entry)
        db.session.commit()
        app.logger.info(f"Timeline entry saved for user {user_id}")
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Failed to save timeline: {e}")


def get_medication_timeline(user_id, limit=10):
    try:
        timeline = (
            MedicationTimeline.query
            .filter_by(user_id=user_id)
            .order_by(MedicationTimeline.timestamp.desc())
            .limit(limit)
            .all()
        )

        timeline_data = [{
            'timestamp': t.timestamp.isoformat(),
            'age': t.patient_age,
            'med_count': t.medication_count,
            'ade_risk': round(t.ade_risk * 100, 1),
            'beers_count': t.beers_count,
            'ddi_count': t.ddi_count,
            'medications': t.medications_snapshot
        } for t in reversed(timeline)]

        return timeline_data
    except Exception as e:
        app.logger.error(f"Error fetching timeline: {e}")
        return []


# CONTEXT MEMORY
def get_conversation_context(user_id, limit=5):
    try:
        recent_chats = (
            ChatHistory.query
            .filter_by(user_id=user_id)
            .order_by(ChatHistory.timestamp.desc())
            .limit(limit)
            .all()
        )

        context = []
        for chat in reversed(recent_chats):
            context.append(f"User: {chat.message[:200]}")
            context.append(f"Assistant: {chat.response[:200]}")

        return "\n".join(context) if context else ""
    except Exception as e:
        app.logger.error(f"Error fetching context: {e}")
        return ""


# FEATURE EXTRACTION
def extract_clinical_features_gemini(user_input: str, context: str = "") -> dict:
    default_features = {
        'Age': None,
        'Gender': 'Female',
        'Creatinine': 1.0,
        'Chronic_Disease_Index': 2,
        'MedicationList_Text': ""
    }

    try:
        if gemini_model is None:
            raise Exception("Gemini model not initialized")

        context_prefix = f"Previous context:\n{context}\n\n" if context else ""

        extraction_prompt = f"""{context_prefix}Extract patient information and return ONLY a JSON object.
IMPORTANT RULES:
- If Age is not found, return "Age": null
- If Creatinine is not found, return "Creatinine": 1.0
- If Chronic_Disease_Index is not found, return "Chronic_Disease_Index": 2
- If Gender is not found, return "Gender": "Female"
- Always return MedicationList_Text, even if empty

Required JSON format:
{{
    "Age": <integer or null>,
    "Gender": "Male" or "Female",
    "Creatinine": <float, default 1.0>,
    "Chronic_Disease_Index": <integer, default 2>,
    "MedicationList_Text": "<comma-separated drug names>"
}}

Text to analyze:
{user_input}

Return ONLY valid JSON, no markdown or explanation."""

        response = gemini_model.generate_content(
            extraction_prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )

        raw_output = response.text.strip()
        raw_output = re.sub(
            r'```json\s*|\s*```', '', raw_output, flags=re.IGNORECASE
        ).strip()

        parsed_data = json.loads(raw_output)

        if isinstance(parsed_data, list):
            if parsed_data:
                parsed_data = parsed_data[0]
            else:
                app.logger.warning("Gemini returned empty list")
                return {
                    "error": "No patient data could be extracted from the input"
                }

        features = {
            'Age': parsed_data.get('Age'),
            'Gender': str(parsed_data.get('Gender', 'Female')),
            'Creatinine': float(parsed_data.get('Creatinine') or 1.0),
            'Chronic_Disease_Index': int(
                parsed_data.get('Chronic_Disease_Index') or 2
            ),
            'MedicationList_Text': str(
                parsed_data.get('MedicationList_Text', '')
            )
        }

        if features['Age'] is None:
            app.logger.warning("Age not found in extraction")
            return {
                "error": "Missing Age information. Please provide patient's age (e.g., 'Patient is 75 years old')."
            }

        try:
            features['Age'] = int(features['Age'])
            if features['Age'] < 18 or features['Age'] > 120:
                return {
                    "error": f"Invalid age value: {features['Age']}. Age must be between 18 and 120."
                }
        except (ValueError, TypeError):
            return {
                "error": "Invalid age format. Please provide a numeric age value."
            }

        app.logger.info(
            f"SUCCESS: Gemini extraction successful - Age: {features['Age']}, "
            f"Meds: {len(features['MedicationList_Text'])} chars"
        )
        return features

    except json.JSONDecodeError as e:
        app.logger.error(f"JSON parsing error: {e}")
        return {"error": "Failed to parse AI response. Please try rephrasing your input."}
    except Exception as e:
        app.logger.error(f"ERROR: Gemini extraction failed: {e}")
        return {"error": f"Feature extraction failed: {str(e)}"}


def extract_clinical_features_regex(user_input: str) -> dict:
    features = {
        'Age': None,
        'Gender': 'Female',
        'Creatinine': 1.0,
        'Chronic_Disease_Index': 2,
        'MedicationList_Text': ""
    }

    age_gender_match = re.search(
        r'(?i)(?:patient\s+is\s+a\s+|aged\s+)?(\d{1,3})\s*'
        r'(?:year(?:s)?\s*old|y/o|yo)[,.\s]*(male|female|m|f)?',
        user_input
    )
    if age_gender_match:
        try:
            age_val = int(age_gender_match.group(1))
            if 18 <= age_val <= 120:
                features['Age'] = age_val
            gender_raw = age_gender_match.group(2) or ''
            if gender_raw:
                features['Gender'] = (
                    'Male' if gender_raw.lower().startswith('m') else 'Female'
                )
        except ValueError:
            pass

    creat_match = re.search(
        r'(?i)Creatinine\s*(?:is|:)?\s*([\d.]+)\s*(?:mg/dL)?',
        user_input
    )
    if creat_match:
        try:
            creat_val = float(creat_match.group(1))
            if 0.1 <= creat_val <= 20.0:
                features['Creatinine'] = creat_val
        except ValueError:
            pass

    cdi_match = re.search(
        r'(?i)(?:Chronic Disease Index|CDI)\s*[:=]\s*(\d+)',
        user_input
    )
    if cdi_match:
        try:
            cdi_val = int(cdi_match.group(1))
            if 0 <= cdi_val <= 20:
                features['Chronic_Disease_Index'] = cdi_val
        except ValueError:
            pass

    meds_match = re.search(
        r'(?i)(?:Meds|Medications|Meds List|Meds:)\s*[:\-]?\s*(.*)',
        user_input, re.DOTALL
    )
    if meds_match:
        meds_text = meds_match.group(1).strip()
        match_end = re.search(r'([.!?]\s+)', meds_text)
        if match_end:
            meds_text = meds_text[:match_end.start()].strip()
        features['MedicationList_Text'] = meds_text

    return features


def get_egfr_category(age: int, gender: str, creatinine: float) -> int:
    try:
        creatinine = max(float(creatinine), 0.1)
        creatinine = min(creatinine, 20.0)
        age = min(max(float(age), 18), 120)

        is_male_factor = 1.0 if str(gender).lower() == 'male' else 0.85
        crcl_proxy = ((140.0 - age) * 72.0 * is_male_factor) / creatinine

        if crcl_proxy >= 60:
            return 3
        elif crcl_proxy >= 30:
            return 2
        else:
            return 1
    except Exception as e:
        app.logger.error(f"eGFR calculation error: {e}")
        return 2


def calculate_polypharmacy_features(med_list_text: str):
    try:
        if not med_list_text or not isinstance(med_list_text, str):
            return 0, 0

        med_names_raw = re.split(r'[,\n;]', med_list_text)
        med_names = []

        for item in med_names_raw:
            item = item.strip()
            if not item or len(item) < 3:
                continue

            clean_name = re.sub(
                r'\s+[\d\.]+\s*(?:mg|mcg|IU|tablet|capsule|puffs|mEq|sachet|hr|TID|BID|QD|q\.d\.)?\s*$',
                '', item, flags=re.IGNORECASE
            ).strip()

            clean_name = re.sub(r'\s*\([^)]*\)\s*', '', clean_name).strip()

            if clean_name and len(clean_name) >= 3:
                med_names.append(clean_name)

        med_names = list(set(med_names))

        beers_count = sum(
            1 for drug in med_names
            if any(risk_drug.lower() in drug.lower() for risk_drug in ADE_RISK_DRUGS)
        )

        ddi_count = max(len(med_names) // 3, 0)

        app.logger.info(
            f"Polypharmacy analysis: {len(med_names)} meds, "
            f"{beers_count} Beers, {ddi_count} DDIs"
        )
        return beers_count, ddi_count

    except Exception as e:
        app.logger.error(f"Error in polypharmacy calculation: {e}")
        return 0, 0


def engineer_features(patient_data: dict):
    try:
        if "error" in patient_data:
            raise ValueError(patient_data["error"])

        age = patient_data.get('Age')
        if age is None:
            raise ValueError("Age is required for ML analysis. Please provide patient age.")

        try:
            age = int(age)
            if age < 18 or age > 120:
                raise ValueError(f"Invalid age: {age}. Must be between 18 and 120.")
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid age format: {patient_data.get('Age')}. Must be a number."
            )

        gender = str(patient_data.get('Gender', 'Female'))

        try:
            creatinine = float(patient_data.get('Creatinine', 1.0))
            if creatinine <= 0 or creatinine > 20:
                app.logger.warning(
                    f"Creatinine out of range: {creatinine}, using default 1.0"
                )
                creatinine = 1.0
        except (ValueError, TypeError):
            creatinine = 1.0

        try:
            cdi = int(patient_data.get('Chronic_Disease_Index', 2))
            if cdi < 0 or cdi > 20:
                cdi = 2
        except (ValueError, TypeError):
            cdi = 2

        med_list_text = str(patient_data.get('MedicationList_Text', ''))
        beers_count, ddi_count = calculate_polypharmacy_features(med_list_text)

        med_names = [
            m.strip() for m in re.split(r'[,\n;]', med_list_text) if m.strip()
        ]
        rx_count = len(set(med_names))

        egfr_cat = get_egfr_category(age, gender, creatinine)

        data_row = {
            'age': age,
            'PolypharmacyCount': rx_count,
            'Comorbidity Index': cdi,
            'eGFR Category': egfr_cat,
            'PIM Count': beers_count,
            'DDI Count': ddi_count,
            'Gender_Male': 1 if gender.lower() == 'male' else 0
        }

        features_df = pd.DataFrame([data_row])

        if MODEL_FEATURES:
            features_df = features_df.reindex(columns=MODEL_FEATURES, fill_value=0.0)

        app.logger.info(
            f"Feature engineering successful: Age={age}, Rx={rx_count}, Beers={beers_count}"
        )
        return features_df, med_list_text

    except ValueError as e:
        app.logger.error(f"Feature engineering validation error: {e}")
        raise
    except Exception as e:
        app.logger.error(f"Feature engineering unexpected error: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        raise ValueError(f"Failed to process patient data: {str(e)}")


# RECOMMENDATIONS
def generate_rule_based_recommendations(features_df, ade_prob, eff_score):
    recommendations = []
    try:
        rx_count = int(features_df['PolypharmacyCount'].iloc[0])
        beers_count = int(features_df['PIM Count'].iloc[0])
        egfr_cat = int(features_df['eGFR Category'].iloc[0])
        age = int(features_df['age'].iloc[0])
        cdi = int(features_df['Comorbidity Index'].iloc[0])
        ddi_count = int(features_df['DDI Count'].iloc[0])

        if ade_prob > 0.20:
            recommendations.append(
                "CRITICAL: Very high ADE risk (>20%). Immediate comprehensive medication review required."
            )
        elif ade_prob > 0.15:
            recommendations.append(
                "HIGH RISK: Elevated ADE probability (>15%). Schedule urgent medication review within 48 hours."
            )
        elif ade_prob > 0.08:
            recommendations.append(
                "MODERATE RISK: ADE probability above baseline. Schedule comprehensive review within 7 days."
            )
        else:
            recommendations.append(
                "ACCEPTABLE RISK: Current ADE probability within acceptable range."
            )

        if beers_count >= 3:
            recommendations.append(
                f"MULTIPLE PIMs: {beers_count} potentially inappropriate medications detected. Urgent deprescribing review needed."
            )
        elif beers_count >= 2:
            recommendations.append(
                f"PIM Alert: {beers_count} potentially inappropriate medications per AGS Beers Criteria."
            )
        elif beers_count == 1:
            recommendations.append(
                "PIM Alert: 1 potentially inappropriate medication. Consider safer alternatives."
            )

        if egfr_cat == 1:
            recommendations.append(
                "SEVERE Renal Impairment (eGFR <30): URGENT dose adjustments required."
            )
        elif egfr_cat == 2:
            recommendations.append(
                "Moderate Renal Impairment (eGFR 30-60): Dose adjustments may be needed."
            )

        if rx_count >= 10:
            recommendations.append(
                f"SEVERE Polypharmacy ({rx_count} medications): Initiate structured deprescribing protocol."
            )
        elif rx_count >= 6:
            recommendations.append(
                f"Polypharmacy ({rx_count} medications): Comprehensive medication review recommended."
            )

        if ddi_count >= 3:
            recommendations.append(
                f"HIGH DDI Risk: {ddi_count} potential major drug-drug interactions."
            )
        elif ddi_count >= 1:
            recommendations.append(
                f"DDI Monitoring: {ddi_count} potential drug interactions. Monitor for adverse effects."
            )

        if age >= 85:
            recommendations.append(
                "Advanced elderly (≥85 years): Consider geriatric syndrome screening and fall risk assessment."
            )
        elif age >= 75:
            recommendations.append(
                "Elderly patient (≥75 years): Increase medication monitoring frequency."
            )

        if cdi >= 5:
            recommendations.append(
                f"High comorbidity burden (Index {cdi}): Ensure treatment goals align with patient preferences."
            )

        if eff_score < 0.4:
            recommendations.append(
                "SUBOPTIMAL Effectiveness: Medication regimen optimization strongly recommended."
            )
        elif eff_score < 0.6:
            recommendations.append(
                "Moderate effectiveness: Consider optimization opportunities."
            )

        if len(recommendations) == 1 and "ACCEPTABLE" in recommendations[0]:
            recommendations.append(
                "Continue current regimen with routine monitoring per clinical guidelines."
            )

        return recommendations

    except Exception as e:
        app.logger.error(f"Error generating recommendations: {e}")
        return ["Error generating detailed recommendations. Manual clinical review recommended."]


def generate_shap_guided_recommendations(features_df, ade_prob, eff_score, shap_contributions):
    recommendations = []
    try:
        top_risk_factors = shap_contributions[
            shap_contributions['Contribution'] > 0
        ].nlargest(3, 'Abs')
        protective_factors = shap_contributions[
            shap_contributions['Contribution'] < 0
        ].nsmallest(3, 'Abs')

        rx_count = int(features_df['PolypharmacyCount'].iloc[0])
        beers_count = int(features_df['PIM Count'].iloc[0])
        egfr_cat = int(features_df['eGFR Category'].iloc[0])
        age = int(features_df['age'].iloc[0])
        ddi_count = int(features_df['DDI Count'].iloc[0])
        cdi = int(features_df['Comorbidity Index'].iloc[0])

        if ade_prob > 0.20:
            recommendations.append(
                "CRITICAL: Very high ADE risk (>20%) driven by SHAP-identified risk factors. Immediate intervention required."
            )
        elif ade_prob > 0.15:
            recommendations.append(
                "HIGH RISK: Elevated ADE probability (>15%). Urgent medication review within 48 hours."
            )
        elif ade_prob > 0.08:
            recommendations.append(
                "MODERATE RISK: ADE probability above baseline. Schedule review within 7 days."
            )
        else:
            recommendations.append(
                "ACCEPTABLE RISK: Current ADE probability within acceptable range."
            )

        for _, row in top_risk_factors.iterrows():
            feature = row['Feature']
            contribution = row['Contribution']
            impact_pct = abs(contribution) * 100

            if feature == 'PIM Count' and beers_count > 0:
                recommendations.append(
                    f"PIM Count (SHAP impact: {impact_pct:.1f}%) - "
                    f"{beers_count} potentially inappropriate medication(s) detected. "
                    f"Initiate deprescribing review per AGS Beers Criteria."
                )
            elif feature == 'age' and age >= 75:
                recommendations.append(
                    f"Advanced age (SHAP impact: {impact_pct:.1f}%) - "
                    f"Patient age {age} significantly increases vulnerability. "
                    f"Implement geriatric syndrome screening and fall risk assessment."
                )
            elif feature == 'PolypharmacyCount' and rx_count >= 6:
                recommendations.append(
                    f"Polypharmacy burden (SHAP impact: {impact_pct:.1f}%) - "
                    f"{rx_count} medications increase interaction risk. "
                    f"Conduct comprehensive medication reconciliation."
                )
            elif feature == 'eGFR Category' and egfr_cat <= 2:
                recommendations.append(
                    f"Renal impairment (SHAP impact: {impact_pct:.1f}%) - "
                    f"eGFR Category {egfr_cat} requires immediate dose adjustments."
                )
            elif feature == 'DDI Count' and ddi_count >= 1:
                recommendations.append(
                    f"Drug interactions (SHAP impact: {impact_pct:.1f}%) - "
                    f"{ddi_count} potential DDIs detected. Review medication combinations."
                )
            elif feature == 'Comorbidity Index':
                recommendations.append(
                    f"Comorbidity burden (SHAP impact: {impact_pct:.1f}%) - "
                    f"Comorbidity Index {cdi} increases complexity. "
                    f"Align treatment goals with patient preferences."
                )

        if len(protective_factors) > 0:
            top_protective = protective_factors.iloc[0]
            feature = top_protective['Feature']
            impact_pct = abs(top_protective['Contribution']) * 100
            recommendations.append(
                f"PROTECTIVE FACTOR: {feature} (SHAP impact: -{impact_pct:.1f}%) "
                f"is helping reduce risk. Maintain current management strategy."
            )

        if eff_score < 0.4:
            recommendations.append(
                "SUBOPTIMAL Effectiveness: SHAP analysis suggests significant optimization potential."
            )
        elif eff_score < 0.6:
            recommendations.append(
                "Moderate effectiveness: SHAP model indicates room for improvement."
            )

        if len(recommendations) == 1:
            recommendations.append(
                "Continue current regimen with routine monitoring per clinical guidelines."
            )

        return recommendations[:6]

    except Exception as e:
        app.logger.error(f"Error in SHAP-guided recommendations: {e}")
        return generate_rule_based_recommendations(features_df, ade_prob, eff_score)


def get_xai_and_recommendation(features_df, ade_prob, eff_score, med_list_text):
    if ADE_EXPLAINER is None:
        app.logger.warning("SHAP explainer not available, using rule-based recommendations")
        return generate_rule_based_recommendations(features_df, ade_prob, eff_score), []

    try:
        features_numeric = features_df.copy()
        feature_names = list(features_numeric.columns)

        for col in features_numeric.columns:
            try:
                features_numeric[col] = pd.to_numeric(features_numeric[col], errors='coerce')
            except Exception as e:
                app.logger.error(f"Error converting column {col}: {e}")
                features_numeric[col] = 0.0

        features_numeric = features_numeric.fillna(0.0)
        features_numeric = features_numeric.replace([np.inf, -np.inf], 0.0)
        features_numeric = features_numeric.astype(np.float64)

        features_array = features_numeric.values.astype(np.float64)

        app.logger.info("Calling SHAP explainer...")
        shap_values_obj = ADE_EXPLAINER(features_array)
        app.logger.info("SHAP call successful!")

        if isinstance(shap_values_obj, shap.Explanation):
            shap_values = shap_values_obj.values
            if shap_values.ndim == 3:
                shap_values_class1 = shap_values[0, :, 1]
            elif shap_values.ndim == 2:
                if shap_values.shape[1] == len(feature_names):
                    shap_values_class1 = shap_values[0, :]
                elif shap_values.shape[0] == len(feature_names):
                    shap_values_class1 = (
                        shap_values[:, 1] if shap_values.shape[1] > 1 else shap_values[:, 0]
                    )
                else:
                    shap_values_class1 = shap_values[0, :]
            else:
                shap_values_class1 = shap_values.flatten()[:len(feature_names)]
        else:
            shap_array = np.array(shap_values_obj)
            if shap_array.ndim >= 2:
                shap_values_class1 = shap_array[0, :]
            else:
                shap_values_class1 = shap_array.flatten()

        shap_values_class1 = shap_values_class1[:len(feature_names)]

        shap_contributions = pd.DataFrame({
            'Feature': feature_names,
            'Contribution': shap_values_class1
        })
        shap_contributions['Abs'] = np.abs(shap_contributions['Contribution'])

        top_contributors = (
            shap_contributions[shap_contributions['Abs'] > 0.001]
            .sort_values(by='Abs', ascending=False)
            .head(5)
        )

        recommendations = generate_shap_guided_recommendations(
            features_df, ade_prob, eff_score, shap_contributions
        )

        return recommendations, top_contributors.to_dict('records')

    except Exception as e:
        app.logger.error(f"SHAP analysis error: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        fallback_recs = generate_rule_based_recommendations(features_df, ade_prob, eff_score)
        return fallback_recs, []


# RAG EVIDENCE HELPER
def get_rag_evidence_for_case(patient_data, med_list_text, context, user_input):
    rag = get_rag_system()
    if rag is None or gemini_model is None:
        app.logger.warning("RAG system not available")
        return None

    try:
        age = patient_data.get('Age', 'unknown')
        gender = patient_data.get('Gender', 'Unknown')

        rag_query = (
            f"Patient: {age} years old, {gender}. "
            f"Medications: {med_list_text[:200]}. "
            f"Clinical question: {user_input[:300]}. "
            "Focus on: drug safety in elderly, interactions, renal considerations, deprescribing."
        )

        result = rag.generate_rag_response(
            user_query=rag_query,
            gemini_model=gemini_model,
            conversation_context=context or "",
            top_k=5
        )

        app.logger.info(
            f"RAG: Retrieved {result.get('retrieved_docs', 0)} docs, "
            f"confidence={result.get('confidence', 'unknown')}"
        )

        return result

    except Exception as e:
        app.logger.error(f"RAG evidence error: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        return None


# REPORT GENERATION
def generate_xai_report(patient_data, features_df, ade_prob, eff_score,
                        recommendations, top_contributors, med_list_text, rag_result=None):

    ade_pct = round(ade_prob * 100, 1)
    eff_pct = round(eff_score * 100, 0)

    rx_count = int(features_df['PolypharmacyCount'].iloc[0])
    age = int(patient_data.get('Age', 70))
    gender = patient_data.get('Gender', 'Not specified')
    creatinine = patient_data.get('Creatinine', 'Not specified')
    cdi = int(features_df['Comorbidity Index'].iloc[0])
    beers_count = int(features_df['PIM Count'].iloc[0])
    egfr_cat = int(features_df['eGFR Category'].iloc[0])

    if ade_pct > 15:
        risk_class = "high-risk"
        risk_badge = "high"
        risk_label = "HIGH RISK"
        progress_class = "high"
    elif ade_pct > 8:
        risk_class = "moderate-risk"
        risk_badge = "moderate"
        risk_label = "MODERATE"
        progress_class = "moderate"
    else:
        risk_class = "low-risk"
        risk_badge = "low"
        risk_label = "LOW RISK"
        progress_class = "low"

    if eff_pct >= 60:
        eff_class = "low-risk"
        eff_badge = "low"
        eff_label = "OPTIMAL"
        eff_progress = "low"
    elif eff_pct >= 40:
        eff_class = "moderate-risk"
        eff_badge = "moderate"
        eff_label = "MODERATE"
        eff_progress = "moderate"
    else:
        eff_class = "high-risk"
        eff_badge = "high"
        eff_label = "SUBOPTIMAL"
        eff_progress = "high"

    findings_html = ""
    findings = []

    if ade_pct > 30:
        findings.append(f"Critical ADE risk ({ade_pct}%) identified through SHAP analysis requiring immediate clinical review.")
    elif ade_pct > 15:
        findings.append(f"Elevated ADE risk ({ade_pct}%) with specific modifiable factors identified by SHAP.")
    else:
        findings.append(f"Acceptable ADE risk ({ade_pct}%) with SHAP analysis identifying key contributing factors.")

    if top_contributors and len(top_contributors) > 0:
        for contributor in top_contributors[:3]:
            feature = contributor.get('Feature', '')
            contribution = contributor.get('Contribution', 0)
            impact_pct = abs(contribution) * 100

            if contribution > 0:
                if feature == 'PIM Count' and beers_count > 0:
                    findings.append(
                        f"SHAP identifies PIM Count (impact: {impact_pct:.1f}%) as key driver: "
                        f"{beers_count} potentially inappropriate medications elevate risk."
                    )
                elif feature == 'age' and age >= 75:
                    findings.append(
                        f"Advanced age (SHAP impact: {impact_pct:.1f}%): "
                        f"Patient age {age} significantly increases vulnerability."
                    )
                elif feature == 'PolypharmacyCount' and rx_count >= 6:
                    findings.append(
                        f"Polypharmacy burden (SHAP impact: {impact_pct:.1f}%): "
                        f"{rx_count} medications increase interaction complexity."
                    )

    for finding in findings[:5]:
        findings_html += (
            '<div class="finding-item"><div class="bullet"></div>'
            f'<div class="finding-text">{finding}</div></div>\n'
        )

    rec_html = ""
    for i, rec in enumerate(recommendations[:5], 1):
        clean_rec = re.sub(r'[🚨⚠️📈📋💊👴🏥✅🔴]', '', rec).strip()
        clean_rec = clean_rec.replace('**', '').strip()
        clean_rec = re.sub(
            r'^(CRITICAL|ELEVATED RISK|OPTIMAL|PRIORITY|HIGH RISK|ACCEPTABLE RISK):\s*',
            '', clean_rec
        )
        rec_html += (
            f'<div class="recommendation-item">'
            f'<div class="rec-number">{i}</div>'
            f'<div class="rec-text">{clean_rec}</div></div>\n'
        )

    rag_section_html = ""
    if rag_result and rag_result.get("retrieved_docs", 0) > 0:
        rag_text = rag_result.get("response", "").strip()
        rag_sources = rag_result.get("sources", [])
        confidence = rag_result.get("confidence", "unknown").upper()

        rag_bullets_html = ""
        if rag_text:
            lines = rag_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•')):
                    clean_line = re.sub(r'^[-•]\s*', '', line).strip()
                    if clean_line:
                        rag_bullets_html += (
                            '<div class="rag-bullet-item">'
                            '<div class="rag-bullet-dot"></div>'
                            f'<div class="rag-bullet-text">{clean_line}</div>'
                            '</div>\n'
                        )

        table_rows = ""
        for idx, src in enumerate(rag_sources, 1):
            table_rows += (
                "<tr>"
                f"<td>{idx}</td>"
                f"<td>{src.get('source', 'Unknown')}</td>"
                f"<td>{src.get('topic', 'N/A')}</td>"
                f"<td>{src.get('category', 'N/A')}</td>"
                "</tr>\n"
            )

        rag_section_html = f"""
        <div class="section">
            <div class="section-header">Evidence-Based Clinical Guidance</div>
            <div class="rag-info-box">
                <strong>Retrieval-Augmented Generation (RAG)</strong> retrieved relevant clinical evidence from the internal medical knowledge base.
                <div style="margin-top:8px">
                    <span class="confidence-badge confidence-{confidence.lower()}">{confidence} CONFIDENCE</span>
                    <span style="margin-left:12px;color:#64748b">Retrieved: <strong>{len(rag_sources)}</strong> clinical references</span>
                </div>
            </div>
            <div class="rag-card">
                <div class="rag-card-title">Clinical Evidence Summary</div>
                <div class="rag-bullets-container">
                    {rag_bullets_html if rag_bullets_html else '<div class="rag-bullet-text" style="color:#64748b">No specific evidence bullets generated</div>'}
                </div>
            </div>
            <div class="rag-sources-card">
                <div class="rag-card-title">Knowledge Base Sources</div>
                <table class="rag-table">
                    <thead>
                        <tr><th style="width:50px">#</th><th>Source</th><th>Topic</th><th>Category</th></tr>
                    </thead>
                    <tbody>
                        {table_rows if table_rows else '<tr><td colspan="4" style="text-align:center;color:#64748b">No sources available</td></tr>'}
                    </tbody>
                </table>
            </div>
        </div>
        """

    report_html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
.medical-report {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #ffffff; max-width: 1000px; margin: 0 auto; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
.report-header {{ background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white; padding: 40px; text-align: center; border-bottom: 4px solid #1e40af; }}
.report-title {{ font-size: 32px; font-weight: 700; margin-bottom: 8px; }}
.report-subtitle {{ font-size: 16px; opacity: 0.95; font-weight: 500; }}
.report-date {{ margin-top: 16px; font-size: 14px; opacity: 0.9; }}
.report-body {{ padding: 40px; }}
.section {{ margin-bottom: 40px; }}
.section-header {{ font-size: 20px; font-weight: 700; color: #1e3a8a; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 3px solid #3b82f6; }}
.patient-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 30px; }}
.stat-card {{ background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #3b82f6; text-align: center; }}
.stat-value {{ font-size: 36px; font-weight: 800; color: #1e293b; margin-bottom: 6px; }}
.stat-label {{ font-size: 13px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
.risk-cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 30px; }}
.risk-card {{ background: white; border: 2px solid #e2e8f0; border-radius: 16px; padding: 28px; }}
.risk-card.high-risk {{ border-color: #ef4444; background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%); }}
.risk-card.low-risk {{ border-color: #10b981; background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%); }}
.risk-card.moderate-risk {{ border-color: #f59e0b; background: linear-gradient(135deg, #fffbeb 0%, #ffffff 100%); }}
.risk-header {{ font-size: 14px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }}
.risk-score {{ font-size: 56px; font-weight: 800; color: #1e293b; line-height: 1; margin-bottom: 12px; }}
.risk-badge {{ display: inline-block; padding: 8px 16px; border-radius: 20px; font-size: 13px; font-weight: 700; color: white; margin-bottom: 16px; }}
.risk-badge.high {{ background: #ef4444; }}
.risk-badge.moderate {{ background: #f59e0b; }}
.risk-badge.low {{ background: #10b981; }}
.progress-bar {{ width: 100%; height: 12px; background: #e2e8f0; border-radius: 10px; overflow: hidden; margin-top: 12px; }}
.progress-fill {{ height: 100%; border-radius: 10px; }}
.progress-fill.high {{ background: linear-gradient(90deg, #ef4444, #dc2626); }}
.progress-fill.moderate {{ background: linear-gradient(90deg, #f59e0b, #d97706); }}
.progress-fill.low {{ background: linear-gradient(90deg, #10b981, #059669); }}
.clinical-findings {{ background: #f8fafc; border-radius: 12px; padding: 24px; border: 2px solid #e2e8f0; }}
.finding-item {{ display: flex; align-items: flex-start; gap: 12px; padding: 14px; background: white; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid #3b82f6; }}
.finding-item:last-child {{ margin-bottom: 0; }}
.bullet {{ width: 8px; height: 8px; background: #3b82f6; border-radius: 50%; margin-top: 6px; flex-shrink: 0; }}
.finding-text {{ font-size: 15px; line-height: 1.6; color: #334155; }}
.recommendations {{ background: white; border: 2px solid #e2e8f0; border-radius: 12px; overflow: hidden; }}
.recommendation-item {{ display: flex; align-items: flex-start; gap: 16px; padding: 20px 24px; border-bottom: 1px solid #e2e8f0; }}
.recommendation-item:last-child {{ border-bottom: none; }}
.rec-number {{ display: flex; align-items: center; justify-content: center; min-width: 32px; height: 32px; background: linear-gradient(135deg, #3b82f6, #1e40af); color: white; border-radius: 50%; font-weight: 700; font-size: 15px; flex-shrink: 0; }}
.rec-text {{ font-size: 15px; line-height: 1.6; color: #334155; padding-top: 4px; }}
.rag-info-box {{ background:#eff6ff;border:2px solid #3b82f6;border-radius:10px;padding:16px;margin-bottom:20px;font-size:14px;color:#1e3a8a;line-height:1.6; }}
.confidence-badge {{ display:inline-block;padding:4px 12px;border-radius:12px;font-size:11px;font-weight:700;text-transform:uppercase; }}
.confidence-high {{ background:#10b981;color:white; }}
.confidence-medium {{ background:#f59e0b;color:white; }}
.confidence-low {{ background:#64748b;color:white; }}
.rag-card {{ background:#f8fafc;border:2px solid #e2e8f0;border-radius:12px;padding:20px;margin-bottom:20px; }}
.rag-sources-card {{ background:white;border:2px solid #e2e8f0;border-radius:12px;padding:20px; }}
.rag-card-title {{ font-size:16px;font-weight:700;color:#1e3a8a;margin-bottom:16px;padding-bottom:8px;border-bottom:2px solid #3b82f6; }}
.rag-bullets-container {{ display:flex;flex-direction:column;gap:10px; }}
.rag-bullet-item {{ display:flex;align-items:flex-start;gap:12px;padding:10px 12px;background:white;border-radius:8px;border-left:3px solid #3b82f6; }}
.rag-bullet-dot {{ width:8px;height:8px;border-radius:50%;background:#3b82f6;margin-top:6px;flex-shrink:0; }}
.rag-bullet-text {{ font-size:14px;color:#334155;line-height:1.6; }}
.rag-table {{ width:100%;border-collapse:collapse;font-size:13px;margin-top:12px; }}
.rag-table th {{ background:#eff6ff;color:#1e3a8a;font-weight:600;text-align:left;padding:10px;border-bottom:2px solid #3b82f6; }}
.rag-table td {{ padding:10px;border-bottom:1px solid #e5e7eb;color:#334155; }}
.disclaimer {{ background: linear-gradient(135deg, #fef3c7, #fde68a); border: 2px solid #fbbf24; border-radius: 12px; padding: 20px; margin-top: 30px; }}
.disclaimer-text {{ font-size: 14px; color: #78350f; line-height: 1.6; }}
.model-info {{ text-align: center; padding: 20px; background: #f8fafc; border-top: 2px solid #e2e8f0; margin-top: 20px; }}
.model-badges {{ display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; }}
.model-badge {{ padding: 6px 14px; background: linear-gradient(135deg, #1e3a8a, #3b82f6); color: white; border-radius: 20px; font-size: 12px; font-weight: 600; text-transform: uppercase; }}
</style>
</head>
<body>
<div class="medical-report">
    <div class="report-header">
        <div class="report-title">Clinical Risk Analysis Report</div>
        <div class="report-subtitle">AI-Powered Geriatric Polypharmacy Assessment with SHAP & RAG</div>
        <div class="report-date">{datetime.now().strftime('%B %d, %Y at %H:%M')}</div>
    </div>
    <div class="report-body">
        <div class="section">
            <div class="section-header">Patient Demographics</div>
            <div class="patient-grid">
                <div class="stat-card"><div class="stat-value">{age}</div><div class="stat-label">Years Old</div></div>
                <div class="stat-card"><div class="stat-value">{gender}</div><div class="stat-label">Gender</div></div>
                <div class="stat-card"><div class="stat-value">{rx_count}</div><div class="stat-label">Medications</div></div>
                <div class="stat-card"><div class="stat-value">{creatinine}</div><div class="stat-label">Creatinine</div></div>
                <div class="stat-card"><div class="stat-value">{cdi}</div><div class="stat-label">Comorbidity</div></div>
                <div class="stat-card"><div class="stat-value">Cat {egfr_cat}</div><div class="stat-label">eGFR</div></div>
            </div>
        </div>
        <div class="section">
            <div class="section-header">Risk Assessment</div>
            <div class="risk-cards">
                <div class="risk-card {risk_class}">
                    <div class="risk-header">Adverse Drug Event Risk</div>
                    <div class="risk-score">{ade_pct}<span style="font-size:28px;">%</span></div>
                    <span class="risk-badge {risk_badge}">{risk_label}</span>
                    <div class="progress-bar"><div class="progress-fill {progress_class}" style="width:{min(ade_pct,100)}%;"></div></div>
                </div>
                <div class="risk-card {eff_class}">
                    <div class="risk-header">Treatment Effectiveness</div>
                    <div class="risk-score">{eff_pct}<span style="font-size:28px;">%</span></div>
                    <span class="risk-badge {eff_badge}">{eff_label}</span>
                    <div class="progress-bar"><div class="progress-fill {eff_progress}" style="width:{eff_pct}%;"></div></div>
                </div>
            </div>
        </div>
        <div class="section">
            <div class="section-header">Clinical Rationale (SHAP Analysis)</div>
            <div class="clinical-findings">{findings_html}</div>
        </div>
        <div class="section">
            <div class="section-header">SHAP-Guided Recommendations</div>
            <div class="recommendations">{rec_html}</div>
        </div>
        {rag_section_html}
        <div class="disclaimer">
            <div class="disclaimer-text"><strong>Medical Disclaimer:</strong> This AI-generated report provides clinical decision support only. All medical decisions must be made by qualified healthcare professionals considering complete clinical context.</div>
        </div>
        <div class="model-info">
            <div class="model-badges">
                <span class="model-badge">ML: Logistic Regression</span>
                <span class="model-badge">XAI: SHAP</span>
                <span class="model-badge">RAG: Medical KB</span>
                <span class="model-badge">LLM: Gemini 2.5 Flash</span>
            </div>
        </div>
    </div>
</div>
</body>
</html>'''

    return report_html


# AUTH ROUTES
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Please provide both email and password.', 'danger')
            return redirect(url_for('login'))

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['logged_in'] = True
            session['user_id'] = user.id
            session['user_email'] = user.email
            flash('Login successful! Welcome back.', 'success')
            return redirect(url_for('chat_page'))
        else:
            flash('Invalid email or password. Please try again.', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not email or not password:
            flash('Please provide both email and password.', 'danger')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email address already registered.', 'danger')
            return redirect(url_for('register'))

        try:
            new_user = User(email=email)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Registration error: {e}")
            flash('An error occurred during registration. Please try again.', 'danger')

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))


# PAGES
@app.route('/chat_page')
def chat_page():
    if not session.get('logged_in'):
        flash('Please log in to access the chatbot.', 'warning')
        return redirect(url_for('login'))
    return render_template('index.html')


@app.route('/settings')
def settings_page():
    if not session.get('logged_in'):
        flash('Please log in to access settings.', 'warning')
        return redirect(url_for('login'))
    return render_template('settings.html')


# HISTORY
@app.route('/clear-history', methods=['POST'])
def clear_chat_history():
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401

    try:
        user_id = session.get('user_id')
        ChatHistory.query.filter_by(user_id=user_id).delete()
        db.session.commit()
        flash('Chat history cleared successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error clearing history: {e}")
        flash('Error clearing chat history.', 'danger')

    return redirect(url_for('settings_page'))


@app.route('/api/timeline', methods=['GET'])
def get_timeline_api():
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401

    try:
        user_id = session.get('user_id')
        timeline_data = get_medication_timeline(user_id, limit=20)
        return jsonify({
            "success": True,
            "timeline": timeline_data,
            "count": len(timeline_data)
        })
    except Exception as e:
        app.logger.error(f"Error fetching timeline: {e}")
        return jsonify({"error": "Failed to fetch timeline"}), 500


@app.route('/api/history', methods=['GET'])
def get_chat_history_api():
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401

    try:
        user_id = session.get('user_id')
        limit = request.args.get('limit', 50, type=int)

        history = (
            ChatHistory.query
            .filter_by(user_id=user_id)
            .order_by(ChatHistory.timestamp.desc())
            .limit(limit)
            .all()
        )

        history_data = [{
            'id': h.id,
            'message': h.message,
            'response': h.response,
            'type': h.message_type,
            'timestamp': h.timestamp.isoformat()
        } for h in reversed(history)]

        return jsonify({"success": True, "history": history_data})
    except Exception as e:
        app.logger.error(f"Error fetching history: {e}")
        return jsonify({"error": "Failed to fetch history"}), 500


# MAIN CHAT
@app.route('/chat', methods=['POST'])
def handle_chat_query():
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized. Please log in."}), 401

    raw_data = request.get_json() or {}
    user_input = raw_data.get('query', '').strip()
    chat_tone = raw_data.get('chat_tone', 'clinical')
    image_data = raw_data.get('image_data', None)
    user_id = session.get('user_id')

    app.logger.info(
        f"Chat request from user {user_id}: {len(user_input)} chars, Image: {bool(image_data)}"
    )

    if not gemini_model:
        return jsonify({"error": "AI service unavailable. Please check GEMINI_API_KEY."}), 503

    if not user_input and not image_data:
        return jsonify({"response": "Please enter a query or upload an image."}), 400

    context = get_conversation_context(user_id, limit=3)

    try:
        patient_data = None
        structured_fields = False

        if isinstance(raw_data, dict):
            for key in ('Age', 'age', 'Creatinine', 'creatinine', 'medication_list', 'MedicationList_Text'):
                if raw_data.get(key) is not None:
                    structured_fields = True
                    break

        if structured_fields:
            med_text = ""
            meds_field = (
                raw_data.get('medication_list') or
                raw_data.get('medications') or
                raw_data.get('MedicationList_Text')
            )
            if isinstance(meds_field, list):
                med_text = ", ".join([
                    f"{m.get('drugName','')} {m.get('dosage','')}".strip()
                    for m in meds_field if m
                ])
            elif isinstance(meds_field, str):
                med_text = meds_field

            patient_data = {
                'Age': raw_data.get('age') or raw_data.get('Age'),
                'Gender': raw_data.get('gender') or raw_data.get('Gender') or 'Female',
                'Creatinine': raw_data.get('Creatinine') or raw_data.get('creatinine'),
                'Chronic_Disease_Index': raw_data.get('Chronic_Disease_Index') or raw_data.get('cdi') or 2,
                'MedicationList_Text': med_text
            }
        else:
            patient_data = extract_clinical_features_regex(user_input)

            age_extracted = patient_data.get('Age') is not None
            meds_extracted = bool(patient_data.get('MedicationList_Text', '').strip())

            if not age_extracted or not meds_extracted:
                app.logger.info("Attempting Gemini extraction")
                patient_data = extract_clinical_features_gemini(user_input, context)

        features_df, med_list_text = engineer_features(patient_data)

        age_val = features_df['age'].iloc[0] if 'age' in features_df.columns else patient_data.get('Age')
        age_valid = (age_val is not None and int(age_val) >= 65)
        meds_present = bool(str(med_list_text).strip()) or features_df['PolypharmacyCount'].iloc[0] > 0

        proceed_xai = (age_valid and meds_present) or structured_fields

        if proceed_xai and ADE_MODEL is not None and EFFECTIVENESS_MODEL is not None:
            app.logger.info("Running XAI analysis with ML models")

            ade_prob = float(ADE_MODEL.predict_proba(features_df)[0][1])
            eff_score_raw = float(EFFECTIVENESS_MODEL.predict(features_df)[0])
            eff_score = float(np.clip(eff_score_raw, 0.1, 1.0))

            recommendations, top_contributors = get_xai_and_recommendation(
                features_df, ade_prob, eff_score, med_list_text
            )

            rag_result = get_rag_evidence_for_case(patient_data, med_list_text, context, user_input)

            report_html = generate_xai_report(
                patient_data, features_df, ade_prob, eff_score,
                recommendations, top_contributors, med_list_text,
                rag_result=rag_result
            )

            save_medication_timeline(
                user_id=user_id,
                patient_age=int(features_df['age'].iloc[0]),
                med_count=int(features_df['PolypharmacyCount'].iloc[0]),
                ade_risk=ade_prob,
                beers=int(features_df['PIM Count'].iloc[0]),
                ddi=int(features_df['DDI Count'].iloc[0]),
                meds_text=med_list_text
            )

            try:
                chat_entry = ChatHistory(
                    user_id=user_id,
                    message=user_input[:500],
                    response=f"XAI Analysis Report (ADE Risk: {round(ade_prob*100, 1)}%)",
                    message_type='analysis'
                )
                db.session.add(chat_entry)
                db.session.commit()
            except Exception as e:
                app.logger.error(f"Failed to save chat history: {e}")

            structured_result = {
                'ADE_Risk_Percentage': round(ade_prob * 100, 2),
                'Effectiveness_Score': round(eff_score * 100, 2),
                'Recommendations': recommendations,
                'Top_Contributors': top_contributors,
                'Engineered_Features': features_df.iloc[0].to_dict(),
                'RAG_Used': bool(rag_result and rag_result.get("retrieved_docs", 0) > 0)
            }

            return jsonify({
                'success': True,
                'response': report_html,
                'structured': structured_result,
                'analysis_type': 'xai'
            })
        else:
            missing = []
            if not age_valid:
                missing.append("Age (must be ≥65 years)")
            if not meds_present:
                missing.append("Medication list")
            app.logger.info(f"Insufficient data for ML analysis. Missing: {', '.join(missing)}")

    except Exception as e:
        app.logger.error(f"XAI processing error: {e}")
        import traceback
        app.logger.error(traceback.format_exc())

    # Fallback conversational mode
    app.logger.info("Using conversational mode (no ML analysis)")

    recent_history = []
    try:
        chats = (
            ChatHistory.query
            .filter_by(user_id=user_id)
            .order_by(ChatHistory.timestamp.desc())
            .limit(4)
            .all()
        )
        for chat in reversed(chats):
            recent_history.append({"role": "user", "parts": [chat.message[:300]]})
            recent_history.append({"role": "model", "parts": [chat.response[:300]]})
    except Exception as e:
        app.logger.error(f"History fetch error: {e}")

    tone_prompts = {
        'supportive': (
            "You are a warm, supportive medical assistant specializing in geriatric care. "
            "Use empathetic language while maintaining clinical accuracy."
        ),
        'detailed': (
            "You are a detailed medical educator specializing in geriatric pharmacology. "
            "Provide comprehensive explanations with relevant clinical context."
        ),
        'clinical': (
            "You are a professional clinical assistant specializing in geriatric polypharmacy. "
            "Provide concise, evidence-based information in a professional tone."
        )
    }

    system_prompt = tone_prompts.get(chat_tone, tone_prompts['clinical'])
    system_prompt += (
        "\n\nIMPORTANT: For structured risk analysis reports, users must provide:\n"
        "- Patient age (65+ years)\n"
        "- Complete medication list with dosages\n"
        "- Optional: Creatinine level, gender, comorbidities\n\n"
        "Example: 'Patient is 75 years old, male. Medications: Warfarin 5mg daily, "
        "Metoprolol 50mg BID, Omeprazole 20mg daily. Creatinine: 1.5 mg/dL'\n\n"
        "For general questions or image analysis, provide helpful educational information."
    )

    if context:
        system_prompt += f"\n\nRecent conversation context:\n{context}"

    try:
        chat_parts = []

        if image_data:
            try:
                if ',' in image_data:
                    image_data = image_data.split(',')[1]

                image_bytes = base64.b64decode(image_data)
                pil_image = Image.open(BytesIO(image_bytes))

                chat_parts.append(system_prompt)
                chat_parts.append(pil_image)

                if user_input:
                    chat_parts.append(f"User's question about this medical image: {user_input}")
                else:
                    chat_parts.append(
                        "Analyze this medical image and provide relevant clinical observations. "
                        "Focus on any text, medications, lab values, or clinical information visible."
                    )

            except Exception as e:
                app.logger.error(f"Image processing error: {e}")
                return jsonify({
                    "error": f"Image processing failed: {str(e)}. Please ensure you're uploading a valid image."
                }), 400
        else:
            chat_parts = [system_prompt, user_input]

        response = gemini_model.generate_content(
            chat_parts,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1500
            )
        )

        response_text = response.text.strip()

        try:
            chat_entry = ChatHistory(
                user_id=user_id,
                message=(user_input[:500] if user_input else "[Image uploaded]"),
                response=response_text[:1000],
                message_type='chat'
            )
            db.session.add(chat_entry)
            db.session.commit()
        except Exception as e:
            app.logger.error(f"Failed to save chat history: {e}")

        return jsonify({
            "success": True,
            "response": response_text,
            "analysis_type": "conversational",
            "image_analyzed": bool(image_data)
        })

    except Exception as e:
        error_msg = str(e)

        if "429" in error_msg or "Resource exhausted" in error_msg:
            return jsonify({
                "error": "API rate limit reached. Please wait 60 seconds and try again.",
                "retry_after": 60
            }), 429

        if "SAFETY" in error_msg.upper() or "blocked" in error_msg.lower():
            return jsonify({
                "error": "Content was blocked by safety filters. Please rephrase your query.",
                "type": "safety_block"
            }), 400

        app.logger.error(f"Chat generation error: {e}")
        import traceback
        app.logger.error(traceback.format_exc())

        return jsonify({
            "error": f"Unable to process request: {str(e)}",
            "type": "processing_error"
        }), 500


if __name__ == "__main__":
    print("\n🚀 Starting Polypharmacy Clinical Decision Support System...")
    initialize_rag_once()
    print("\n✅ Server running at: http://127.0.0.1:5000")
    print("🔴 Press CTRL+C to stop\n")
    from waitress import serve
    serve(app, host='127.0.0.1', port=5000)