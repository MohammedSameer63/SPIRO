import { type Prediction } from "../api/reports";

interface PredictionDisplayProps {
  prediction: Prediction | null;
  loading?: boolean;
}

const WASTE_STREAM_COLORS: Record<string, { bg: string; text: string; icon: string }> = {
  organic: { bg: "#dcfce7", text: "#166534", icon: "🌱" },
  recoverable: { bg: "#dbeafe", text: "#1e40af", icon: "♻️" },
  non_recoverable: { bg: "#fef3c7", text: "#92400e", icon: "🗑️" },
  hazardous: { bg: "#fee2e2", text: "#991b1b", icon: "⚠️" },
  sanitary: { bg: "#f3e8ff", text: "#7e22ce", icon: "🏥" },
  e_waste: { bg: "#e5e7eb", text: "#374151", icon: "🔋" },
  reject: { bg: "#f3f4f6", text: "#4b5563", icon: "❓" },
};

function getStreamFromCategory(categoryName: string): string {
  // Categories stored in the DB are disposal streams ("Recoverable",
  // "Non-Recoverable", "E-Waste", ...) — match those exactly first.
  const normalized = categoryName
    .toLowerCase()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const exact: Record<string, string> = {
    organic: "organic",
    recoverable: "recoverable",
    "non recoverable": "non_recoverable",
    hazardous: "hazardous",
    sanitary: "sanitary",
    "e waste": "e_waste",
    reject: "reject",
    unknown: "reject",
  };
  if (exact[normalized]) return exact[normalized];

  // Fallback: keyword match for fine-grained class names.
  const name = normalized;
  if (name.includes("organic") || name.includes("food") || name.includes("garden") || name.includes("leaf") || name.includes("compost")) return "organic";
  if (name.includes("recoverable") || name.includes("recyclable") || name.includes("plastic") || name.includes("glass") || name.includes("metal") || name.includes("paper") || name.includes("cardboard") || name.includes("pizza")) return "recoverable";
  if (name.includes("hazardous") || name.includes("chemical") || name.includes("paint") || name.includes("aerosol")) return "hazardous";
  if (name.includes("sanitary") || name.includes("medical") || name.includes("diaper") || name.includes("mask") || name.includes("syringe") || name.includes("hygiene")) return "sanitary";
  if (name.includes("e waste") || name.includes("electronic") || name.includes("battery") || name.includes("cable") || name.includes("phone") || name.includes("circuit")) return "e_waste";
  if (name.includes("reject") || name.includes("unknown")) return "reject";
  return "non_recoverable";
}

function toPercent(confidence: number | string | null | undefined): number | null {
  if (confidence === null || confidence === undefined) return null;
  const num = typeof confidence === "string" ? Number(confidence) : confidence;
  if (!Number.isFinite(num)) return null;
  return num > 1 ? num : num * 100;
}

function formatConfidence(confidence: number | string | null | undefined): string {
  const pct = toPercent(confidence);
  if (pct === null) return "—";
  return `${pct.toFixed(1)}%`;
}

function getConfidenceColor(confidence: number | string | null | undefined): string {
  const pct = toPercent(confidence);
  if (pct === null) return "#6b7280";
  if (pct >= 80) return "#166534";
  if (pct >= 60) return "#1e40af";
  if (pct >= 40) return "#92400e";
  return "#991b1b";
}

function formatPredictionTime(value: string | null | undefined): string {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString("en-IN", {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function PredictionDisplay({ prediction, loading = false }: PredictionDisplayProps) {
  if (loading) {
    return (
      <div style={styles.loading}>
        <style>{`@keyframes spiro-spin { to { transform: rotate(360deg); } }`}</style>
        <div style={styles.spinner}></div>
        <p>Analyzing waste image...</p>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div style={styles.noPrediction}>
        <p>No AI prediction available yet.</p>
        <p style={styles.hint}>ML analysis runs automatically after report submission.</p>
      </div>
    );
  }

  const categoryName =
    prediction.category_name ||
    (prediction as unknown as { category?: { name?: string } }).category?.name ||
    "Unknown";
  const unclassified =
    prediction.model_used === "yolo_no_detection";
  const stream = unclassified
    ? "reject"
    : getStreamFromCategory(categoryName);
  const streamInfo = WASTE_STREAM_COLORS[stream] || WASTE_STREAM_COLORS.reject;
  const confidence = prediction.confidence;
  const confidencePct = toPercent(confidence);

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>🤖 AI Waste Classification</h3>
        <span style={{
          ...styles.streamBadge,
          background: streamInfo.bg,
          color: streamInfo.text,
        }}>
          {streamInfo.icon} {stream.replace("_", " ").toUpperCase()}
        </span>
      </div>

      <div style={styles.grid}>
        <div style={styles.card}>
          <div style={styles.label}>Detected Category</div>
          <div style={styles.value}>
            {unclassified
              ? "Unclassified"
              : categoryName.replace(/_/g, " ")}
          </div>
        </div>

        <div style={styles.card}>
          <div style={styles.label}>Confidence</div>
          {unclassified ? (
            <div
              style={{
                ...styles.value,
                color: "#6b7280",
                fontSize: "24px",
              }}
            >
              —
            </div>
          ) : (
            <>
              <div
                style={{
                  ...styles.value,
                  color: getConfidenceColor(confidence),
                  fontSize: "24px",
                }}
              >
                {formatConfidence(confidence)}
              </div>
              <div style={styles.confidenceBar}>
                <div
                  style={{
                    ...styles.confidenceFill,
                    width: `${Math.min(100, confidencePct ?? 0)}%`,
                    background: getConfidenceColor(confidence),
                  }}
                ></div>
              </div>
            </>
          )}
        </div>

        <div style={styles.card}>
          <div style={styles.label}>Model Used</div>
          <div style={styles.valueSmall}>{prediction.model_used || "SPIRO Pipeline"}</div>
        </div>

        <div style={styles.card}>
          <div style={styles.label}>Analyzed At</div>
          <div style={styles.valueSmall}>
            {formatPredictionTime(prediction.prediction_time)}
          </div>
        </div>
      </div>

      <div style={styles.disposalGuide}>
        <h4 style={styles.guideTitle}>📋 Disposal Guidance</h4>
        <p style={styles.guideText}>
          {unclassified
            ? "The AI couldn't identify the waste in this photo. A worker will verify the category during collection."
            : getDisposalGuidance(stream)}
        </p>
      </div>
    </div>
  );
}

function getDisposalGuidance(stream: string): string {
  const guides: Record<string, string> = {
    organic: "Place in green/compost bin. Suitable for composting or biogas generation.",
    recoverable: "Place in blue/recycling bin. Ensure items are clean and dry before recycling.",
    non_recoverable: "Place in black/general waste bin. Not suitable for recycling.",
    hazardous: "Take to hazardous waste collection point. Do NOT place in regular bins.",
    sanitary: "Seal in bag and place in sanitary/red bin. Do NOT recycle.",
    e_waste: "Take to e-waste collection center. Contains valuable recoverable materials.",
    reject: "Requires manual inspection. Contact waste management for guidance.",
  };
  return guides[stream] || guides.reject;
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    maxWidth: "800px",
    padding: "24px",
    background: "white",
    borderRadius: "12px",
    boxShadow: "0 4px 15px rgba(0,0,0,0.08)",
    border: "1px solid #e5e7eb",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "20px",
    paddingBottom: "16px",
    borderBottom: "1px solid #e5e7eb",
  },
  title: {
    margin: 0,
    fontSize: "18px",
    fontWeight: "600",
    color: "#1f2937",
  },
  streamBadge: {
    padding: "6px 14px",
    borderRadius: "20px",
    fontSize: "12px",
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: "0.5px",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
    gap: "16px",
    marginBottom: "20px",
  },
  card: {
    padding: "16px",
    background: "#f9fafb",
    borderRadius: "8px",
    border: "1px solid #e5e7eb",
  },
  label: {
    fontSize: "11px",
    fontWeight: "600",
    color: "#6b7280",
    textTransform: "uppercase",
    letterSpacing: "0.5px",
    marginBottom: "6px",
  },
  value: {
    fontSize: "16px",
    fontWeight: "600",
    color: "#1f2937",
  },
  valueSmall: {
    fontSize: "13px",
    fontWeight: "500",
    color: "#374151",
  },
  confidenceBar: {
    marginTop: "8px",
    height: "6px",
    background: "#e5e7eb",
    borderRadius: "3px",
    overflow: "hidden",
  },
  confidenceFill: {
    height: "100%",
    borderRadius: "3px",
    transition: "width 0.3s ease",
  },
  disposalGuide: {
    padding: "16px",
    background: "#f0fdf4",
    borderRadius: "8px",
    border: "1px solid #bbf7d0",
  },
  guideTitle: {
    margin: "0 0 8px 0",
    fontSize: "14px",
    fontWeight: "600",
    color: "#166534",
  },
  guideText: {
    margin: 0,
    fontSize: "13px",
    color: "#166534",
    lineHeight: 1.5,
  },
  loading: {
    display: "flex",
    flexDirection: "column" as const,
    alignItems: "center",
    padding: "40px 20px",
    color: "#6b7280",
    gap: "12px",
  },
  spinner: {
    width: "24px",
    height: "24px",
    border: "3px solid #e5e7eb",
    borderTopColor: "#15803d",
    borderRadius: "50%",
    animation: "spiro-spin 1s linear infinite",
  },
  noPrediction: {
    padding: "24px",
    textAlign: "center" as const,
    color: "#6b7280",
    background: "#f9fafb",
    borderRadius: "8px",
    border: "1px dashed #d1d5db",
  },
  hint: {
    fontSize: "13px",
    marginTop: "8px",
    color: "#9ca3af",
  },
};