"""
Waterborne Disease Prediction API
FastAPI backend serving the multimodal BiLSTM model for disease classification.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
import joblib  # type: ignore
import numpy as np  # type: ignore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model artefact paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_multimodal_model.h5")
TOKENIZER_PATH = os.path.join(SCRIPT_DIR, "tokenizer.joblib")
SCALER_PATH = os.path.join(SCRIPT_DIR, "scaler.joblib")
LABEL_ENCODER_PATH = os.path.join(SCRIPT_DIR, "label_encoder.joblib")

MAX_SEQ_LEN = 60  # must match training configuration

# ---------------------------------------------------------------------------
# Global model references (populated during startup)
# ---------------------------------------------------------------------------
model = None
tokenizer = None
scaler = None
label_encoder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artefacts on startup, release on shutdown."""
    global model, tokenizer, scaler, label_encoder

    logger.info("Loading model artefacts …")
    try:
        model = load_model(MODEL_PATH)
        tokenizer = joblib.load(TOKENIZER_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        logger.info("All artefacts loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load model artefacts: %s", exc)

    yield  # application runs

    logger.info("Shutting down …")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Waterborne Disease Prediction API",
    description="Multimodal BiLSTM-based prediction of waterborne diseases from symptoms and lab values.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    symptoms: str = Field(..., min_length=3, description="Free-text patient symptom description")
    sodium: float = Field(135.0, ge=0, description="Sodium (mmol/L)")
    potassium: float = Field(4.1, ge=0, description="Potassium (mmol/L)")
    chloride: float = Field(98.0, ge=0, description="Chloride (mmol/L)")
    wbc: float = Field(8.0, ge=0, description="WBC count (×10⁹/L)")
    hemoglobin: float = Field(13.0, ge=0, description="Hemoglobin (g/dL)")
    platelets: float = Field(250.0, ge=0, description="Platelets (×10⁹/L)")
    urea: float = Field(13.5, ge=0, description="Urea (mg/dL)")
    creatinine: float = Field(0.8, ge=0, description="Creatinine (mg/dL)")
    bilirubin: float = Field(0.7, ge=0, description="Bilirubin (mg/dL)")
    alt: float = Field(25.0, ge=0, description="ALT (U/L)")
    ast: float = Field(28.0, ge=0, description="AST (U/L)")
    age: int = Field(35, ge=0, le=150, description="Patient age (years)")
    gender: str = Field("Male", description="Patient gender (Male / Female)")
    hygiene: int = Field(5, ge=1, le=10, description="Hygiene score (1–10)")
    water_source: str = Field("Tap", description="Primary water source (Tap / Well / River / Bottled)")


class DiseaseProbability(BaseModel):
    disease: str
    probability: str
    score: float


class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    probabilities: list[DiseaseProbability]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    """Quick health-check endpoint."""
    models_loaded = all([model, tokenizer, scaler, label_encoder])
    return {"status": "ok" if models_loaded else "degraded", "models_loaded": models_loaded}


@app.post("/predict", response_model=PredictionResponse)
def predict_disease(req: PredictionRequest):
    """Run inference on patient data and return predicted disease with probabilities."""
    if not all([model, tokenizer, scaler, label_encoder]):
        raise HTTPException(status_code=503, detail="Models are not loaded. Please check server logs.")

    # --- Text branch ---
    seq = tokenizer.texts_to_sequences([req.symptoms])
    seq_pad = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding="post")

    # --- Numerical features (13 values, same order as training) ---
    lab_values = np.array([[
        req.sodium, req.potassium, req.chloride, req.wbc, req.hemoglobin,
        req.platelets, req.urea, req.creatinine, req.bilirubin, req.alt,
        req.ast, req.age, req.hygiene,
    ]])
    lab_scaled = scaler.transform(lab_values)

    # --- Categorical one-hot features ---
    gender_male = 1 if req.gender == "Male" else 0
    water_bottled = 1 if req.water_source == "Bottled" else 0
    water_river = 1 if req.water_source == "River" else 0
    water_well = 1 if req.water_source == "Well" else 0
    cat_features = np.array([[gender_male, water_bottled, water_river, water_well]])

    final_features = np.hstack([lab_scaled, cat_features])

    # --- Predict ---
    pred = model.predict([seq_pad, final_features], verbose=0)
    disease_idx = int(np.argmax(pred))
    disease_name = str(label_encoder.inverse_transform([disease_idx])[0])
    confidence = float(pred[0][disease_idx] * 100)

    # --- Build sorted probability list ---
    prob_data = []
    for i, prob_val in enumerate(pred[0]):
        d_name = str(label_encoder.inverse_transform([i])[0])
        prob_data.append(DiseaseProbability(
            disease=d_name,
            probability=f"{prob_val * 100:.2f}%",
            score=float(prob_val),
        ))
    prob_data.sort(key=lambda x: x.score, reverse=True)

    return PredictionResponse(disease=disease_name, confidence=confidence, probabilities=prob_data)
