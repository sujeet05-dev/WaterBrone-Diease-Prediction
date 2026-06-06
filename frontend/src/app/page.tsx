"use client";

import { useState, useEffect, useRef } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
type DiseaseProbability = {
  disease: string;
  probability: string;
  score: number;
};

type PredictionResponse = {
  disease: string;
  confidence: number;
  probabilities: DiseaseProbability[];
};

type ExtractedField = {
  value: string | number;
  source: "extracted" | "default";
};

type ImagePredictionResponse = {
  disease: string;
  confidence: number;
  probabilities: DiseaseProbability[];
  extracted_data: Record<string, ExtractedField>;
};

type InputMode = "manual" | "image";

// ---------------------------------------------------------------------------
// Scroll-reveal wrapper (IntersectionObserver)
// ---------------------------------------------------------------------------
function FadeInSection({ children }: { children: React.ReactNode }) {
  const [isVisible, setVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          obs.unobserve(el);
        }
      },
      { threshold: 0.1 }
    );
    obs.observe(el);
    return () => obs.unobserve(el);
  }, []);

  return (
    <div className={`fade-in-section ${isVisible ? "is-visible" : ""}`} ref={ref}>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Default form values (normal healthy ranges)
// ---------------------------------------------------------------------------
const DEFAULT_FORM = {
  symptoms: "",
  sodium: 135.0,
  potassium: 4.1,
  chloride: 98.0,
  wbc: 8.0,
  hemoglobin: 13.0,
  platelets: 250.0,
  urea: 13.5,
  creatinine: 0.8,
  bilirubin: 0.7,
  alt: 25.0,
  ast: 28.0,
  age: 35,
  gender: "Male",
  hygiene: 5,
  water_source: "Tap",
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Friendly labels for extracted fields
// ---------------------------------------------------------------------------
const FIELD_LABELS: Record<string, string> = {
  symptoms: "Symptoms",
  sodium: "Sodium (mmol/L)",
  potassium: "Potassium (mmol/L)",
  chloride: "Chloride (mmol/L)",
  wbc: "WBC (×10⁹/L)",
  hemoglobin: "Hemoglobin (g/dL)",
  platelets: "Platelets (×10⁹/L)",
  urea: "Urea (mg/dL)",
  creatinine: "Creatinine (mg/dL)",
  bilirubin: "Bilirubin (mg/dL)",
  alt: "ALT (U/L)",
  ast: "AST (U/L)",
  age: "Age",
  gender: "Gender",
  hygiene: "Hygiene Score",
  water_source: "Water Source",
};

export default function Home() {
  // Shared state
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [inputMode, setInputMode] = useState<InputMode>("manual");

  // Manual entry state
  const [formData, setFormData] = useState(DEFAULT_FORM);

  // Image upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [extractedData, setExtractedData] = useState<Record<string, ExtractedField> | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => {
      if (["symptoms", "gender", "water_source"].includes(name)) {
        return { ...prev, [name]: value };
      }
      if (["age", "hygiene"].includes(name)) {
        return { ...prev, [name]: Math.round(Number(value)) || 0 };
      }
      return { ...prev, [name]: Number(value) || 0 };
    });
  };

  const handleReset = () => {
    setFormData(DEFAULT_FORM);
    setResult(null);
    setError(null);
    setSelectedFile(null);
    setImagePreview(null);
    setExtractedData(null);
  };

  // --- Mode switch ---
  const switchMode = (mode: InputMode) => {
    setInputMode(mode);
    setResult(null);
    setError(null);
    setExtractedData(null);
  };

  // --- Manual submit ---
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (formData.symptoms.trim().length < 3) {
      setError("Please describe symptoms in at least 3 characters before running the analysis.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        let errorMsg = `Server responded with ${res.status}`;
        if (body?.detail) {
          errorMsg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
        }
        throw new Error(errorMsg);
      }
      const data: PredictionResponse = await res.json();
      setResult(data);
      setTimeout(() => document.getElementById("results")?.scrollIntoView({ behavior: "smooth" }), 300);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setError(`Prediction failed — ${message}. Is the FastAPI server running on port 8000?`);
    } finally {
      setLoading(false);
    }
  };

  // --- Image file handling ---
  const processFile = (file: File) => {
    const allowed = ["image/jpeg", "image/png", "image/webp", "image/jpg"];
    if (!allowed.includes(file.type)) {
      setError("Unsupported file type. Please upload a JPEG, PNG, or WebP image.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError("Image is too large. Maximum size is 10 MB.");
      return;
    }
    setError(null);
    setSelectedFile(file);
    setExtractedData(null);
    setResult(null);
    const reader = new FileReader();
    reader.onload = (ev) => setImagePreview(ev.target?.result as string);
    reader.readAsDataURL(file);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) processFile(e.target.files[0]);
  };

  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(false); };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.[0]) processFile(e.dataTransfer.files[0]);
  };

  const removeImage = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setExtractedData(null);
    setResult(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // --- Image submit ---
  const handleImageSubmit = async () => {
    if (!selectedFile) {
      setError("Please select or drop an image first.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    setExtractedData(null);

    try {
      const fd = new FormData();
      fd.append("file", selectedFile);
      const res = await fetch("http://127.0.0.1:8000/predict-from-image", {
        method: "POST",
        body: fd,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        let errorMsg = `Server responded with ${res.status}`;
        if (body?.detail) {
          errorMsg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
        }
        throw new Error(errorMsg);
      }
      const data: ImagePredictionResponse = await res.json();
      setResult({ disease: data.disease, confidence: data.confidence, probabilities: data.probabilities });
      setExtractedData(data.extracted_data);
      setTimeout(() => document.getElementById("extracted-data")?.scrollIntoView({ behavior: "smooth" }), 300);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setError(`Image prediction failed — ${message}. Make sure the FastAPI server is running and GEMINI_API_KEY is set.`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* ----- Hero ----- */}
      <section className="hero">
        <h1 className="title-main">Waterborne Disease Predictor</h1>
        <p>
          Combining Advanced Deep Learning, Natural Language Processing, and
          Bio-Chemical markers to deliver an early and highly accurate disease
          diagnosis.
        </p>
        <a
          href="#prediction-form"
          className="scroll-indicator"
          onClick={(e) => {
            e.preventDefault();
            document.getElementById("prediction-form")?.scrollIntoView({ behavior: "smooth" });
          }}
        >
          <span>Scan Patient Now</span>
          <div className="chevron" />
        </a>
      </section>

      <div className="container" id="prediction-form" style={{ paddingTop: "5rem" }}>
        {/* ----- Mode Tabs ----- */}
        <div className="mode-tabs">
          <button
            className={`mode-tab ${inputMode === "manual" ? "active" : ""}`}
            onClick={() => switchMode("manual")}
          >
            ✏️ Manual Entry
          </button>
          <button
            className={`mode-tab ${inputMode === "image" ? "active" : ""}`}
            onClick={() => switchMode("image")}
          >
            📷 Upload Report Image
          </button>
        </div>

        {/* ----- Manual Prediction Form ----- */}
        {inputMode === "manual" && (
        <FadeInSection>
          <form onSubmit={handleSubmit} className="glass-card">
            {/* Symptoms */}
            <h2 className="section-title">🩺 Patient Symptoms</h2>
            <div className="form-group">
              <label htmlFor="symptoms">Describe the symptoms in detail:</label>
              <textarea
                id="symptoms"
                name="symptoms"
                className="form-control"
                placeholder="Example: Severe watery diarrhea for 2 days, vomiting, extreme dehydration, weakness, and mild fever."
                value={formData.symptoms}
                onChange={handleChange}
                required
              />
            </div>

            {/* Electrolytes & Blood */}
            <h2 className="section-title" style={{ marginTop: "3rem" }}>⚡ Electrolytes &amp; Blood Chemistry</h2>
            <div className="form-grid">
              {[
                { label: "Sodium (mmol/L)", name: "sodium", step: 0.1 },
                { label: "Potassium (mmol/L)", name: "potassium", step: 0.1 },
                { label: "Chloride (mmol/L)", name: "chloride", step: 0.1 },
                { label: "WBC Count (×10⁹/L)", name: "wbc", step: 0.1 },
                { label: "Hemoglobin (g/dL)", name: "hemoglobin", step: 0.1 },
                { label: "Platelets (×10⁹/L)", name: "platelets", step: 0.1 },
              ].map((f) => (
                <div className="form-group" key={f.name}>
                  <label>{f.label}</label>
                  <input
                    type="number"
                    step={f.step}
                    name={f.name}
                    className="form-control"
                    value={(formData as Record<string, unknown>)[f.name] as number}
                    onChange={handleChange}
                  />
                </div>
              ))}
            </div>

            {/* Kidney & Liver */}
            <h2 className="section-title" style={{ marginTop: "3rem" }}>🔬 Kidney &amp; Liver Function</h2>
            <div className="form-grid">
              {[
                { label: "Urea (mg/dL)", name: "urea", step: 0.1 },
                { label: "Creatinine (mg/dL)", name: "creatinine", step: 0.1 },
                { label: "Bilirubin (mg/dL)", name: "bilirubin", step: 0.1 },
                { label: "ALT (U/L)", name: "alt", step: 0.1 },
                { label: "AST (U/L)", name: "ast", step: 0.1 },
              ].map((f) => (
                <div className="form-group" key={f.name}>
                  <label>{f.label}</label>
                  <input
                    type="number"
                    step={f.step}
                    name={f.name}
                    className="form-control"
                    value={(formData as Record<string, unknown>)[f.name] as number}
                    onChange={handleChange}
                  />
                </div>
              ))}
            </div>

            {/* Patient Info */}
            <h2 className="section-title" style={{ marginTop: "3rem" }}>👤 Patient Information</h2>
            <div className="form-grid">
              <div className="form-group">
                <label>Age (years)</label>
                <input type="number" name="age" className="form-control" value={formData.age} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>Gender</label>
                <select name="gender" className="form-control" value={formData.gender} onChange={handleChange}>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>
              <div className="form-group">
                <label>Hygiene Score (1–10)</label>
                <input type="range" name="hygiene" min={1} max={10} className="form-control" style={{ padding: 0 }} value={formData.hygiene} onChange={handleChange} />
                <div style={{ textAlign: "center", fontWeight: "bold", marginTop: "0.5rem" }}>
                  {formData.hygiene} / 10
                </div>
              </div>
              <div className="form-group">
                <label>Water Source</label>
                <select name="water_source" className="form-control" value={formData.water_source} onChange={handleChange}>
                  <option value="Tap">Tap</option>
                  <option value="Well">Well</option>
                  <option value="River">River</option>
                  <option value="Bottled">Bottled</option>
                </select>
              </div>
            </div>

            {/* Error display */}
            {error && (
              <div style={{ color: "#ef4444", padding: "1rem", background: "#ffeef0", borderRadius: 12, marginTop: "2rem", border: "1px solid #ef4444" }}>
                <strong>⚠ </strong> {error}
              </div>
            )}

            {/* Action buttons */}
            <div style={{ display: "flex", gap: "1rem", marginTop: "3rem" }}>
              <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? <div className="loader" /> : "🔍 Start AI Diagnostics"}
              </button>
              <button
                type="button"
                onClick={handleReset}
                style={{
                  padding: "1rem 2rem", border: "2px solid var(--primary-color)",
                  borderRadius: 15, background: "transparent", color: "var(--text-main)",
                  fontWeight: 700, fontSize: "1.1rem", cursor: "pointer", fontFamily: "inherit",
                  transition: "all 0.3s ease", whiteSpace: "nowrap",
                }}
              >
                ↺ Reset
              </button>
            </div>
          </form>
        </FadeInSection>
        )}

        {/* ----- Image Upload Form ----- */}
        {inputMode === "image" && (
        <FadeInSection>
          <div className="glass-card">
            <h2 className="section-title">📷 Upload Lab Report Image</h2>
            <p style={{ marginBottom: "1.5rem", opacity: 0.8, lineHeight: 1.6 }}>
              Upload a photo of a medical lab report or patient form. Our AI (powered by Gemini Vision) will automatically
              extract symptoms, lab values, and patient information — then predict the disease instantly.
            </p>

            {/* Drop Zone */}
            <div
              className={`drop-zone ${isDragging ? "dragging" : ""} ${imagePreview ? "has-image" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => !imagePreview && fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={handleFileSelect}
                style={{ display: "none" }}
                id="image-upload"
              />

              {imagePreview ? (
                <div className="image-preview-wrapper">
                  <img src={imagePreview} alt="Lab report preview" className="image-preview" />
                  <button
                    type="button"
                    className="remove-image-btn"
                    onClick={(e) => { e.stopPropagation(); removeImage(); }}
                    title="Remove image"
                  >
                    ✕
                  </button>
                  <div className="image-file-name">
                    📄 {selectedFile?.name} ({((selectedFile?.size || 0) / 1024).toFixed(1)} KB)
                  </div>
                </div>
              ) : (
                <div className="drop-zone-content">
                  <div className="drop-zone-icon">📤</div>
                  <p className="drop-zone-text">Drag & drop your lab report image here</p>
                  <p className="drop-zone-subtext">or click to browse files</p>
                  <p className="drop-zone-formats">Supported: JPEG, PNG, WebP · Max 10 MB</p>
                </div>
              )}
            </div>

            {/* Error display */}
            {error && (
              <div style={{ color: "#ef4444", padding: "1rem", background: "#ffeef0", borderRadius: 12, marginTop: "2rem", border: "1px solid #ef4444" }}>
                <strong>⚠ </strong> {error}
              </div>
            )}

            {/* Action buttons */}
            <div style={{ display: "flex", gap: "1rem", marginTop: "2rem" }}>
              <button
                type="button"
                className="btn-primary"
                disabled={loading || !selectedFile}
                onClick={handleImageSubmit}
              >
                {loading ? <div className="loader" /> : "🔬 Analyse Report & Predict"}
              </button>
              <button
                type="button"
                onClick={handleReset}
                style={{
                  padding: "1rem 2rem", border: "2px solid var(--primary-color)",
                  borderRadius: 15, background: "transparent", color: "var(--text-main)",
                  fontWeight: 700, fontSize: "1.1rem", cursor: "pointer", fontFamily: "inherit",
                  transition: "all 0.3s ease", whiteSpace: "nowrap",
                }}
              >
                ↺ Reset
              </button>
            </div>
          </div>
        </FadeInSection>
        )}

        {/* ----- Extracted Data (Image mode) ----- */}
        {extractedData && (
          <FadeInSection>
            <div id="extracted-data" className="glass-card">
              <h2 className="section-title">🔎 Extracted Data from Report</h2>
              <p style={{ marginBottom: "1.5rem", opacity: 0.8 }}>
                The following values were extracted from the uploaded image. Fields marked as
                <span className="badge badge-extracted" style={{ marginLeft: 6 }}>Extracted</span> were found in the report;
                <span className="badge badge-default" style={{ marginLeft: 6 }}>Default</span> values were used where data was not visible.
              </p>
              <div className="extracted-grid">
                {Object.entries(extractedData).map(([key, field]) => (
                  <div key={key} className={`extracted-item ${field.source}`}>
                    <div className="extracted-label">{FIELD_LABELS[key] || key}</div>
                    <div className="extracted-value">
                      {typeof field.value === "string" && field.value.length > 60
                        ? field.value.substring(0, 60) + "…"
                        : String(field.value)}
                    </div>
                    <span className={`badge badge-${field.source}`}>
                      {field.source === "extracted" ? "✓ Extracted" : "⊘ Default"}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </FadeInSection>
        )}

        {/* ----- Results ----- */}
        {result && (
          <FadeInSection>
            <div id="results" className="glass-card results-card">
              <h2 style={{ fontSize: "2rem" }}>Predicted Disease</h2>
              <div className="disease-name">{result.disease}</div>
              <p style={{ fontSize: "1.4rem", fontWeight: 700, color: "#38bdf8" }}>
                Confidence: {result.confidence.toFixed(2)}%
              </p>

              <div className="prob-list">
                <h3 style={{ textAlign: "left", marginTop: "2rem", fontSize: "1.5rem" }}>
                  📊 Detailed Probability Analysis
                </h3>
                {result.probabilities.map((prob, i) => (
                  <div key={i} className="prob-item">
                    <div style={{ width: 160, textAlign: "left", fontWeight: 700, fontSize: "1.05rem" }}>
                      {prob.disease}
                    </div>
                    <div className="prob-bar-bg">
                      <div className="prob-bar-fill" style={{ width: `${Math.max(prob.score * 100, 0.5)}%` }} />
                    </div>
                    <div style={{ width: 80, textAlign: "right", fontWeight: "bold" }}>
                      {prob.probability}
                    </div>
                  </div>
                ))}
              </div>

              <div
                style={{
                  marginTop: "3rem", padding: "1.5rem",
                  background: "rgba(239,68,68,0.1)", borderLeft: "6px solid #ef4444",
                  borderRadius: 8, textAlign: "left",
                }}
              >
                <strong style={{ color: "#ef4444", fontSize: "1.2rem" }}>⚕️ Medical Disclaimer:</strong>
                <br /><br />
                This prediction is generated by an AI model and should be used as a <strong>screening tool only</strong>.
                Always consult qualified healthcare professionals for proper diagnosis, treatment, and emergency interventions.
              </div>
            </div>
          </FadeInSection>
        )}

        {/* ----- Architecture Info ----- */}
        <section className="info-section">
          <FadeInSection>
            <h2 className="title-main" style={{ fontSize: "3.5rem", textAlign: "center" }}>
              Behind the Architecture
            </h2>
          </FadeInSection>

          <div className="info-grid">
            <FadeInSection>
              <div className="info-card">
                <h3>🧠 Deep Neural Network</h3>
                <p>
                  A <strong>Multimodal Bidirectional LSTM</strong> that processes free-text
                  symptoms via NLP embeddings and merges them with 17 dense clinical
                  features for high-fidelity classification.
                </p>
                <ul>
                  <li>Embedding + BiLSTM text branch</li>
                  <li>13 scaled numerical lab values</li>
                  <li>4 one-hot encoded categorical features</li>
                </ul>
              </div>
            </FadeInSection>

            <FadeInSection>
              <div className="info-card">
                <h3>🎯 Performance Metrics</h3>
                <p>
                  Trained on 10,000 clinically realistic patient records with
                  stratified 80 / 20 train-test split and class-weight balancing.
                </p>
                <ul>
                  <li><strong>Accuracy:</strong> 97.83 % validation</li>
                  <li><strong>Classes:</strong> 9 (8 diseases + Healthy)</li>
                  <li><strong>Inference:</strong> &lt; 200 ms per prediction</li>
                </ul>
              </div>
            </FadeInSection>

            <FadeInSection>
              <div className="info-card">
                <h3>📊 Detectable Diseases</h3>
                <p>
                  The model classifies patient data into one of the following waterborne
                  disease categories:
                </p>
                <ul>
                  <li>Cholera · Typhoid · Dysentery</li>
                  <li>Hepatitis A · Giardiasis</li>
                  <li>E. Coli · Cryptosporidiosis</li>
                  <li>Shigellosis · Healthy</li>
                </ul>
              </div>
            </FadeInSection>
          </div>
        </section>

        {/* ----- Footer ----- */}
        <footer
          style={{
            textAlign: "center", padding: "3rem 1rem 2rem",
            opacity: 0.6, fontSize: "0.95rem",
          }}
        >
          <p>💧 Waterborne Disease Prediction System · © {new Date().getFullYear()}</p>
          <p style={{ marginTop: "0.5rem" }}>
            <strong>Disclaimer:</strong> This tool is for educational and screening purposes only.
            Always consult healthcare professionals.
          </p>
        </footer>
      </div>
    </>
  );
}
