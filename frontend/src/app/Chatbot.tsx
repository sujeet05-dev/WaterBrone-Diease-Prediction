"use client";

import { useState, useRef, useEffect } from "react";

// ---------------------------------------------------------------------------
// Offline keyword → response map.  Order matters — first match wins.
// ---------------------------------------------------------------------------
const knowledgeBase: { keywords: string[]; reply: string }[] = [
  // Greetings
  { keywords: ["hello", "hi", "hey", "good morning", "good evening"],
    reply: "Hello! 👋 I'm your AI Health Assistant. Ask me about waterborne diseases, common symptoms, prevention tips, or when to see a doctor." },

  // Diseases
  { keywords: ["cholera"],
    reply: "Cholera is caused by Vibrio cholerae and spreads through contaminated water. Symptoms include severe watery diarrhea and dehydration. Oral Rehydration Salts (ORS) are essential — please consult a doctor immediately for antibiotics." },
  { keywords: ["typhoid"],
    reply: "Typhoid is caused by Salmonella typhi. Key symptoms are sustained high fever, headache, abdominal pain, and weakness. It requires antibiotic treatment. Please see a healthcare professional." },
  { keywords: ["dysentery"],
    reply: "Dysentery causes bloody diarrhea, stomach cramps, and fever. It can be bacterial (Shigella) or amoebic. Stay hydrated with ORS and visit a clinic as soon as possible." },
  { keywords: ["hepatitis"],
    reply: "Hepatitis A is a viral liver infection spread through contaminated water. Symptoms include jaundice, fatigue, and nausea. There is no specific treatment — supportive care and rest are vital. Consult a doctor." },
  { keywords: ["giardia"],
    reply: "Giardiasis is a parasitic infection causing greasy diarrhea, gas, and stomach cramps. It is treated with anti-parasitic medication — please consult a doctor." },
  { keywords: ["e. coli", "ecoli", "e.coli"],
    reply: "E. coli infections cause severe stomach cramps and often bloody diarrhea. Most cases resolve in a week, but seek medical help immediately if symptoms are severe." },
  { keywords: ["cryptosporidium", "crypto"],
    reply: "Cryptosporidiosis causes watery diarrhea, stomach cramps, and dehydration. It is caused by a parasite resistant to chlorine. Stay hydrated and see a healthcare provider." },
  { keywords: ["shigella", "shigellosis"],
    reply: "Shigellosis causes bloody diarrhea, fever, and stomach cramps. It spreads easily. Antibiotics are often needed — please visit a clinic." },

  // Symptoms
  { keywords: ["fever"],
    reply: "Fever is a common sign of infection. Drink plenty of fluids, rest, and monitor your temperature. If it exceeds 103 °F (39.4 °C) or persists for more than two days, see a doctor." },
  { keywords: ["diarrhea", "diarrhoea", "loose motion"],
    reply: "Diarrhea leads to rapid fluid loss. Take oral rehydration salts (ORS) and drink safe water frequently. If it is bloody, lasts more than 48 hours, or you can't keep fluids down — visit a doctor immediately." },
  { keywords: ["vomit", "nausea", "throwing up"],
    reply: "Vomiting depletes vital fluids and electrolytes. Sip small amounts of ORS frequently. If it persists or you can't keep any fluids down, seek medical help." },
  { keywords: ["headache"],
    reply: "Headaches can result from dehydration, fever, or infection. Stay well hydrated and rest. If severe or accompanied by a stiff neck or confusion, seek emergency care." },
  { keywords: ["stomach", "abdominal", "cramp"],
    reply: "Stomach cramps may signal an intestinal infection. Avoid solid food if severe, stay hydrated, and see a doctor if pain worsens or is accompanied by fever or bloody stools." },
  { keywords: ["dehydrat"],
    reply: "Dehydration is dangerous — signs include extreme thirst, dark urine, dry mouth, and dizziness. Drink ORS immediately. If symptoms are severe, seek emergency care." },
  { keywords: ["jaundice", "yellow"],
    reply: "Yellowing of skin or eyes (jaundice) may indicate a liver issue such as Hepatitis A. Please consult a doctor for proper diagnosis and blood tests." },
  { keywords: ["blood", "bloody"],
    reply: "Blood in stools or vomit is a serious symptom. Stop self-treatment and visit the nearest healthcare facility immediately." },

  // General topics
  { keywords: ["treatment", "cure", "medicine"],
    reply: "General treatment for waterborne diseases involves ORS for hydration and, in many cases, antibiotics or anti-parasitic medicines. Exact treatment depends on the specific disease — always consult a qualified doctor." },
  { keywords: ["prevent", "safe water", "avoid"],
    reply: "Prevention tips: ✅ Drink boiled or purified water. ✅ Wash hands with soap before eating. ✅ Avoid street food in outbreak areas. ✅ Ensure proper sanitation. ✅ Get vaccinated where possible (e.g. Hepatitis A, Typhoid)." },
  { keywords: ["when to see", "doctor", "emergency", "hospital"],
    reply: "Seek immediate medical attention if you experience: high fever (>103 °F), bloody stools or vomit, severe dehydration, confusion, rapid heartbeat, or symptoms lasting more than 2 days." },
  { keywords: ["water source", "tap", "well", "river", "bottled"],
    reply: "Contaminated water is the primary vector. Tap and bottled water are generally safer, while well and river water carry higher risk. Always boil or filter untreated water before drinking." },
  { keywords: ["model", "accuracy", "how does it work", "ai"],
    reply: "Our AI model is a Multimodal Bidirectional LSTM neural network trained on 10,000 patient records. It analyses both symptom text and 17 clinical lab values, achieving 97.83 % validation accuracy across 9 disease classes." },
  { keywords: ["thank", "thanks", "bye", "goodbye"],
    reply: "You're welcome! Stay healthy and remember — for any serious concerns, always consult a qualified doctor. Take care! 👋" },
];

// Default fallback
const FALLBACK =
  "I appreciate your question, but that's beyond my simple knowledge base. For complex or urgent health concerns, I strongly recommend consulting a qualified doctor or visiting your nearest healthcare facility.";

function getReply(userText: string): string {
  const lower = userText.toLowerCase();
  for (const entry of knowledgeBase) {
    if (entry.keywords.some((kw) => lower.includes(kw))) {
      return entry.reply;
    }
  }
  return FALLBACK;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<{ sender: "user" | "bot"; text: string }[]>([
    { sender: "bot", text: "Hi! 👋 I can answer simple questions about waterborne diseases, symptoms, and prevention. How can I help?" },
  ]);
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isOpen]);

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    const userText = input.trim();
    if (!userText) return;

    setMessages((prev) => [...prev, { sender: "user", text: userText }]);
    setInput("");

    // Simulate slight thinking delay
    setTimeout(() => {
      setMessages((prev) => [...prev, { sender: "bot", text: getReply(userText) }]);
    }, 500);
  };

  return (
    <>
      {/* Floating toggle button */}
      <button
        aria-label={isOpen ? "Close chatbot" : "Open chatbot"}
        onClick={() => setIsOpen(!isOpen)}
        style={{
          position: "fixed", bottom: 30, right: 30, width: 65, height: 65,
          borderRadius: "50%", border: "none",
          background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
          display: "flex", justifyContent: "center", alignItems: "center",
          color: "white", fontSize: 28, cursor: "pointer",
          boxShadow: "0 8px 25px rgba(0,0,0,0.3)", zIndex: 1000,
          transition: "transform 0.3s ease, box-shadow 0.3s",
        }}
        onMouseEnter={(e) => (e.currentTarget.style.transform = "scale(1.1)")}
        onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
      >
        {isOpen ? "✕" : "💬"}
      </button>

      {/* Chat window */}
      {isOpen && (
        <div
          style={{
            position: "fixed", bottom: 110, right: 30, width: 370, height: 520,
            background: "var(--bg-color)",
            border: "1px solid var(--glass-border)", borderRadius: 20,
            display: "flex", flexDirection: "column",
            boxShadow: "0 10px 40px rgba(0,0,0,0.2)", zIndex: 1000,
            overflow: "hidden", animation: "slideUpChat 0.3s ease",
          }}
        >
          {/* Header */}
          <div
            style={{
              padding: "1.2rem",
              background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
              color: "white", fontWeight: "bold",
              display: "flex", justifyContent: "space-between", alignItems: "center",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 20 }}>🤖</span>
              <span>AI Health Assistant</span>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              style={{ background: "transparent", border: "none", color: "white", cursor: "pointer", fontSize: 20 }}
            >
              ✕
            </button>
          </div>

          {/* Messages */}
          <div
            style={{
              flex: 1, padding: "1rem", overflowY: "auto",
              display: "flex", flexDirection: "column", gap: 12,
            }}
          >
            {messages.map((m, i) => (
              <div
                key={i}
                style={{
                  alignSelf: m.sender === "user" ? "flex-end" : "flex-start",
                  background:
                    m.sender === "user"
                      ? "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                      : "rgba(102, 126, 234, 0.1)",
                  color: m.sender === "user" ? "white" : "var(--text-main)",
                  padding: "12px 16px", borderRadius: 18, maxWidth: "85%", lineHeight: "1.5",
                  borderBottomRightRadius: m.sender === "user" ? 4 : 18,
                  borderBottomLeftRadius: m.sender === "bot" ? 4 : 18,
                  boxShadow: "0 2px 5px rgba(0,0,0,0.05)",
                  fontSize: "0.95rem",
                }}
              >
                {m.text}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input area */}
          <form
            onSubmit={handleSend}
            style={{
              display: "flex", padding: 15,
              borderTop: "1px solid rgba(102,126,234,0.2)", background: "var(--card-bg)",
            }}
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a health question…"
              style={{
                flex: 1, padding: 12, border: "1px solid rgba(102,126,234,0.3)",
                borderRadius: 25, outline: "none",
                background: "var(--bg-color)", color: "var(--text-main)", fontSize: "1rem",
                fontFamily: "inherit",
              }}
            />
            <button
              type="submit"
              style={{
                background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                border: "none", color: "white", fontSize: 18,
                marginLeft: 10, height: 45, width: 45, borderRadius: "50%",
                cursor: "pointer", display: "flex", justifyContent: "center", alignItems: "center",
                boxShadow: "0 4px 10px rgba(102,126,234,0.3)",
              }}
            >
              ➤
            </button>
          </form>
        </div>
      )}

      <style>{`
        @keyframes slideUpChat {
          from { opacity: 0; transform: translateY(20px) scale(0.95); }
          to   { opacity: 1; transform: translateY(0)    scale(1); }
        }
      `}</style>
    </>
  );
}
