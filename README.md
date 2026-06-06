# 💧 Waterborne Disease Prediction — Full-Stack AI System

## Overview

An AI-powered diagnostic screening tool that predicts **9 waterborne diseases** from patient symptoms and laboratory results. The system combines a **Multimodal Bidirectional LSTM** deep-learning model with a modern **Next.js** frontend, a **FastAPI** backend, and **Groq Vision AI** for automated lab report extraction.

### Detectable Diseases

Cholera · Typhoid · Hepatitis A · Giardiasis · Dysentery · E. Coli Infection · Cryptosporidiosis · Shigellosis · Healthy

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Manual Entry** | Enter patient symptoms, lab values, and demographic info via an interactive form |
| **Image Upload** | Upload a photo of a lab report — AI extracts all values automatically |
| **Groq Vision OCR** | Uses Groq's `meta-llama/llama-4-scout-17b-16e-instruct` model to parse medical reports |
| **Real-time Prediction** | Multimodal BiLSTM model returns disease prediction with confidence scores |
| **Probability Breakdown** | Visual bar chart showing probability distribution across all 9 classes |
| **Drag & Drop** | Drag and drop lab report images directly into the browser |
| **Responsive UI** | Glassmorphism-styled interface with smooth scroll-reveal animations |

---

## 🏗 Architecture

```
┌──────────────────┐       HTTP/JSON       ┌──────────────────┐
│   Next.js  (UI)  │  ←───────────────────→ │  FastAPI (API)   │
│  localhost:3000   │   /predict             │  localhost:8000   │
│                   │   /predict-from-image  │                   │
└──────────────────┘                        └────────┬─────────┘
                                                     │
                                            ┌────────▼─────────┐
                                            │  TensorFlow/Keras │
                                            │  BiLSTM Model     │
                                            │  (.h5 + joblib)   │
                                            └──────────────────┘
                                                     │
                                            ┌────────▼─────────┐
                                            │  Groq Vision API  │
                                            │  (Image → JSON)   │
                                            └──────────────────┘
```

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 16, React 19, TypeScript, Vanilla CSS (glassmorphism) |
| Backend | FastAPI, Uvicorn, Pydantic |
| ML Model | TensorFlow / Keras — Multimodal BiLSTM |
| Image OCR | Groq Vision API (LLaMA 4 Scout) |
| Preprocessing | Scikit-learn (StandardScaler, LabelEncoder), Keras Tokenizer |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** (LTS recommended)
- **Groq API Key** — get one free at [console.groq.com/keys](https://console.groq.com/keys) *(required for image upload feature)*

### 1. Clone the repository

```bash
git clone https://github.com/sujeet05-dev/WaterBrone-Diease-Prediction.git
cd WaterBrone-Diease-Prediction
```

### 2. Start the FastAPI backend

```bash
cd app
pip install -r requirements.txt

# Set the Groq API key (required for /predict-from-image)
# Windows (PowerShell):
$env:GROQ_API_KEY="your_groq_api_key_here"

# Linux / macOS:
export GROQ_API_KEY="your_groq_api_key_here"

uvicorn main:app --reload --port 8000
```

The API will be available at **http://localhost:8000**.
Health check: `GET /health`

### 3. Start the Next.js frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns model load status |
| `POST` | `/predict` | Predict disease from manually entered patient data (JSON body) |
| `POST` | `/predict-from-image` | Upload a lab report image → extract data via Groq Vision → predict disease |

### `/predict` — Request Body

```json
{
  "symptoms": "Severe watery diarrhea, vomiting, dehydration",
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
  "water_source": "Tap"
}
```

### `/predict-from-image` — Multipart Form

Upload a JPEG, PNG, or WebP image (max 10 MB) as a `file` field.

---

## 📁 Project Structure

```
WaterBrone-Diease-Prediction/
├── app/                          # FastAPI backend
│   ├── main.py                   # API server + prediction endpoints
│   ├── requirements.txt          # Python dependencies
│   ├── best_multimodal_model.h5  # Trained Keras model weights
│   ├── tokenizer.joblib          # Text tokenizer
│   ├── scaler.joblib             # StandardScaler for lab values
│   └── label_encoder.joblib      # Disease label encoder
│
├── frontend/                     # Next.js frontend
│   └── src/app/
│       ├── page.tsx              # Main prediction UI (manual + image modes)
│       ├── layout.tsx            # Root layout
│       └── globals.css           # Global styles (glassmorphism theme)
│
├── model/                        # Training data & outputs
│   ├── 2_waterborne_diseases_lab_10k_clean.csv
│   └── cnn_output/               # Alternative CNN model artefacts
│
├── train_cnn.py                  # CNN training script
└── README.md
```

---

## 🧠 Model Details

- **Architecture:** Embedding → BiLSTM (text) + Dense layers (lab values) → Concatenation → Softmax
- **Training data:** 10,000 synthetic but clinically realistic patient records
- **Validation accuracy:** 97.83%
- **Input features:** Free-text symptoms + 13 numerical lab values + 4 one-hot categorical features
- **Inference time:** < 200 ms per prediction

---

## 🖼️ Image Upload Feature

The **Upload Report Image** mode allows users to:

1. **Drag & drop** or browse for a lab report image (JPEG, PNG, WebP)
2. The image is sent to the **Groq Vision API** (`meta-llama/llama-4-scout-17b-16e-instruct`)
3. The AI extracts all 16 fields (symptoms, lab values, demographics) from the image
4. Extracted values are displayed with **Extracted** / **Default** badges
5. Missing fields are filled with healthy-range defaults
6. The prediction model runs on the merged data and returns results

> **Note:** This feature requires the `GROQ_API_KEY` environment variable to be set.

---

## 🔧 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | For image upload | API key from [console.groq.com](https://console.groq.com/keys) |

---

## ⚠️ Disclaimer

This tool is for **educational and screening purposes only**. Always consult qualified healthcare professionals for diagnosis and treatment.
