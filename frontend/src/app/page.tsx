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
export default function Home() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState(DEFAULT_FORM);

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
  };

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
        {/* ----- Prediction Form ----- */}
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
