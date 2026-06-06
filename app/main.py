"""
Waterborne Disease Prediction API
FastAPI backend serving the multimodal BiLSTM model for disease classification.
"""

import os
import io
import json

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
import joblib  # type: ignore
import numpy as np  # type: ignore
import base64
from groq import Groq
from PIL import Image

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
gemini_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artefacts on startup, release on shutdown."""
    global model, tokenizer, scaler, label_encoder, gemini_client

    logger.info("Loading model artefacts …")
    try:
        model = load_model(MODEL_PATH)
        tokenizer = joblib.load(TOKENIZER_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        logger.info("All artefacts loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load model artefacts: %s", exc)

    # Initialize Groq client for image OCR
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if groq_api_key:
        try:
            gemini_client = Groq(api_key=groq_api_key)
            logger.info("Groq Vision client initialized.")
        except Exception as exc:
            logger.error("Failed to initialize Groq client: %s", exc)
    else:
        logger.warning("GROQ_API_KEY not set — /predict-from-image will be unavailable.")

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


class ExtractedField(BaseModel):
    value: str | float | int
    source: str  # "extracted" or "default"


class ImagePredictionResponse(BaseModel):
    disease: str
    confidence: float
    probabilities: list[DiseaseProbability]
    extracted_data: dict[str, ExtractedField]


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


# ---------------------------------------------------------------------------
# Gemini Vision — structured extraction prompt
# ---------------------------------------------------------------------------
GEMINI_EXTRACTION_PROMPT = """You are a medical data extraction assistant. Analyze this image of a medical lab report / patient form and extract the following fields.

Return ONLY a valid JSON object with these exact keys. If a field is not visible or not present in the image, set its value to null.

{
  "symptoms": "<string: free-text description of patient symptoms>",
  "sodium": <number: Sodium in mmol/L>,
  "potassium": <number: Potassium in mmol/L>,
  "chloride": <number: Chloride in mmol/L>,
  "wbc": <number: WBC count in ×10⁹/L>,
  "hemoglobin": <number: Hemoglobin in g/dL>,
  "platelets": <number: Platelets in ×10⁹/L>,
  "urea": <number: Urea / BUN in mg/dL>,
  "creatinine": <number: Creatinine in mg/dL>,
  "bilirubin": <number: Bilirubin in mg/dL>,
  "alt": <number: ALT / SGPT in U/L>,
  "ast": <number: AST / SGOT in U/L>,
  "age": <integer: Patient age in years>,
  "gender": "<string: Male or Female>",
  "hygiene": <integer: Hygiene score 1-10, estimate from context or null>,
  "water_source": "<string: Tap, Well, River, or Bottled — extract or null>"
}

IMPORTANT:
- Extract numbers as raw numeric values, NOT strings.
- For symptoms, combine all mentioned symptoms into a single descriptive sentence.
- Return ONLY the JSON, no markdown fences, no explanation."""

# Default healthy-range values used when a field can't be extracted
DEFAULT_VALUES = {
    "symptoms": "not specified",
    "sodium": 135.0,
    "potassium": 4.1,
    "chloride": 98.0,
    "wbc": 8.0,
    "hemoglobin": 13.0,
    "platelets": 250.0,
    "urea": 13.5,
    "creatinine": 0.8,
    "bilirubin": 0.7,
    "alt": 25.0,
    "ast": 28.0,
    "age": 35,
    "gender": "Male",
    "hygiene": 5,
    "water_source": "Tap",
}


@app.post("/predict-from-image", response_model=ImagePredictionResponse)
async def predict_from_image(file: UploadFile = File(...)):
    """Extract medical data from an uploaded lab report image and predict disease."""

    # --- Validate prerequisites ---
    if not all([model, tokenizer, scaler, label_encoder]):
        raise HTTPException(status_code=503, detail="ML models are not loaded.")
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Groq Vision API is not configured. Set the GROQ_API_KEY environment variable.")

    # --- Validate file type ---
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{file.content_type}'. Upload JPEG, PNG, or WebP.")

    # --- Read image bytes ---
    try:
        image_bytes = await file.read()
        if len(image_bytes) > 10 * 1024 * 1024:  # 10 MB limit
            raise HTTPException(status_code=400, detail="Image too large. Maximum size is 10 MB.")
        image = Image.open(io.BytesIO(image_bytes))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    # --- Encode image as base64 for Groq Vision API ---
    buffered = io.BytesIO()
    image.save(buffered, format=image.format or "PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    mime_type = file.content_type or "image/png"

    # --- Call Groq Vision API ---
    try:
        response = gemini_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": GEMINI_EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{img_b64}"},
                        },
                    ],
                }
            ],
            temperature=0,
        )
        raw_text = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3].strip()
        extracted = json.loads(raw_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail=f"Groq returned invalid JSON. Raw response: {raw_text[:500]}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Groq Vision API error: {exc}")

    # --- Merge extracted data with defaults ---
    extracted_fields: dict[str, ExtractedField] = {}
    final_data: dict = {}

    for key, default_val in DEFAULT_VALUES.items():
        raw_val = extracted.get(key)
        if raw_val is not None and raw_val != "" and str(raw_val).lower() != "null":
            # Coerce to correct type
            try:
                if isinstance(default_val, float):
                    coerced = float(raw_val)
                elif isinstance(default_val, int) and not isinstance(default_val, bool):
                    coerced = int(float(raw_val))
                else:
                    coerced = str(raw_val)
                final_data[key] = coerced
                extracted_fields[key] = ExtractedField(value=coerced, source="extracted")
            except (ValueError, TypeError):
                final_data[key] = default_val
                extracted_fields[key] = ExtractedField(value=default_val, source="default")
        else:
            final_data[key] = default_val
            extracted_fields[key] = ExtractedField(value=default_val, source="default")

    logger.info("Extracted data from image: %s", {k: v.source for k, v in extracted_fields.items()})

    # --- Run prediction (same as /predict) ---
    seq = tokenizer.texts_to_sequences([final_data["symptoms"]])
    seq_pad = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding="post")

    lab_values = np.array([[
        final_data["sodium"], final_data["potassium"], final_data["chloride"],
        final_data["wbc"], final_data["hemoglobin"], final_data["platelets"],
        final_data["urea"], final_data["creatinine"], final_data["bilirubin"],
        final_data["alt"], final_data["ast"], final_data["age"],
        final_data["hygiene"],
    ]])
    lab_scaled = scaler.transform(lab_values)

    gender_male = 1 if final_data["gender"] == "Male" else 0
    water_bottled = 1 if final_data["water_source"] == "Bottled" else 0
    water_river = 1 if final_data["water_source"] == "River" else 0
    water_well = 1 if final_data["water_source"] == "Well" else 0
    cat_features = np.array([[gender_male, water_bottled, water_river, water_well]])

    final_features = np.hstack([lab_scaled, cat_features])

    pred = model.predict([seq_pad, final_features], verbose=0)
    disease_idx = int(np.argmax(pred))
    disease_name = str(label_encoder.inverse_transform([disease_idx])[0])
    confidence = float(pred[0][disease_idx] * 100)

    prob_data = []
    for i, prob_val in enumerate(pred[0]):
        d_name = str(label_encoder.inverse_transform([i])[0])
        prob_data.append(DiseaseProbability(
            disease=d_name,
            probability=f"{prob_val * 100:.2f}%",
            score=float(prob_val),
        ))
    prob_data.sort(key=lambda x: x.score, reverse=True)

    return ImagePredictionResponse(
        disease=disease_name,
        confidence=confidence,
        probabilities=prob_data,
        extracted_data=extracted_fields,
    )
