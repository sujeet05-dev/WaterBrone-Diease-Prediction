# 💧 Waterborne Disease Prediction — Full-Stack AI System

## Overview

An AI-powered diagnostic screening tool that predicts **9 waterborne diseases** from patient symptoms and laboratory results. The system combines a **Multimodal Bidirectional LSTM** deep-learning model with a modern **Next.js** frontend and a **FastAPI** backend.

### Detectable Diseases

Cholera · Typhoid · Hepatitis A · Giardiasis · Dysentery · E. Coli Infection · Cryptosporidiosis · Shigellosis · Healthy

---

## 🏗 Architecture

```
┌──────────────────┐       HTTP/JSON       ┌──────────────────┐
│   Next.js  (UI)  │  ←───────────────────→ │  FastAPI (API)   │
│   localhost:3000  │     /predict endpoint  │  localhost:8000   │
└──────────────────┘                        └────────┬─────────┘
                                                     │
                                            ┌────────▼─────────┐
                                            │  TensorFlow/Keras │
                                            │  BiLSTM Model     │
                                            │  (.h5 + joblib)   │
                                            └──────────────────┘
```

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16, TypeScript, Vanilla CSS (glassmorphism) |
| Backend | FastAPI, Uvicorn, Pydantic |
| ML Model | TensorFlow / Keras — Multimodal BiLSTM |
| Preprocessing | Scikit-learn (StandardScaler, LabelEncoder), Keras Tokenizer |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** (LTS recommended)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/WaterBrone-Diease-Prediction.git
cd WaterBrone-Diease-Prediction
```

### 2. Start the FastAPI backend

```bash
cd app
pip install -r requirements.txt
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

## 📁 Project Structure

```
WaterBrone-Diease-Prediction/
├── app/                          # FastAPI backend
│   ├── main.py                   # API server + prediction endpoint
│   ├── requirements.txt          # Python dependencies
│   ├── best_multimodal_model.h5  # Trained Keras model weights
│   ├── tokenizer.joblib          # Text tokenizer
│   ├── scaler.joblib             # StandardScaler for lab values
│   └── label_encoder.joblib      # Disease label encoder
│
├── frontend/                     # Next.js frontend
│   └── src/app/
│       ├── page.tsx              # Main prediction UI
│       ├── Chatbot.tsx           # AI health assistant (offline)
│       ├── layout.tsx            # Root layout
│       └── globals.css           # Global styles
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
- **Validation accuracy:** 97.83 %
- **Input features:** Free-text symptoms + 13 numerical lab values + 4 one-hot categorical features

---

## 🤖 Chatbot

The frontend includes an **offline AI health assistant** (no API calls required). It answers common questions about:

- Waterborne diseases (Cholera, Typhoid, etc.)
- Symptoms (fever, diarrhea, vomiting, etc.)
- Prevention and when to see a doctor

For complex queries it recommends consulting a healthcare professional.

---

## ⚠️ Disclaimer

This tool is for **educational and screening purposes only**. Always consult qualified healthcare professionals for diagnosis and treatment.
