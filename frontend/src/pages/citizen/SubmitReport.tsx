import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Camera,
  Check,
  ChevronRight,
  Coins,
  ImagePlus,
  TriangleAlert,
} from "lucide-react";
import { createReport, type Report } from "../../api/reports";
import {
  Bar,
  Btn,
  Card,
  Spinner,
  WastePill,
} from "../../components/ui";
import {
  COLORS,
  DISPOSAL_GUIDE,
  DO_NOT_MIX,
  font,
  toStreamKey,
} from "../../theme";

const ESTIMATED_REWARD: Record<string, number> = {
  organic: 12,
  recoverable: 18,
  hazardous: 25,
  "e-waste": 30,
  sanitary: 10,
  "non-recoverable": 8,
  reject: 3,
};

const STEPS = ["Scan", "Analyse", "Review", "Done"];

function SubmitReport() {
  const navigate = useNavigate();
  const fileRef = useRef<HTMLInputElement | null>(null);

  const [step, setStep] = useState(0);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [description, setDescription] = useState("");
  const [cameraOpen, setCameraOpen] = useState(false);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [report, setReport] = useState<Report | null>(null);

  function handleFile(file: File | undefined) {
    if (!file) return;
    setError("");
    setImageFile(file);
    setPreview(URL.createObjectURL(file));
    setStep(1);
  }

  async function handleSubmit() {
    if (!imageFile) return;
    if (!description.trim()) {
      setError("Please describe the waste.");
      return;
    }
    try {
      setSubmitting(true);
      setError("");
      setStep(2); // Analyse (backend runs the real pipeline now)
      const created = await createReport(imageFile, description.trim());
      setReport(created);
      setStep(3);
    } catch (err: any) {
      setError(err?.message || "Failed to submit report.");
      setStep(1);
    } finally {
      setSubmitting(false);
    }
  }

  function reset() {
    setStep(0);
    setImageFile(null);
    setPreview(null);
    setDescription("");
    setReport(null);
    setError("");
  }

  if (step === 3 && report) {
    const pred = report.prediction;
    const stream = toStreamKey(pred?.category_name);
    return (
      <div
        style={{
          fontFamily: font,
          maxWidth: 720,
          padding: "24px 0",
          textAlign: "center",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 16,
        }}
      >
        <span style={{ fontSize: 64 }}>
          <Check size={64} color={COLORS.cGreen} />
        </span>
        <div style={{ fontSize: 22, fontWeight: 800, color: COLORS.g900 }}>
          Report Submitted!
        </div>
        <div
          style={{ fontSize: 14, color: COLORS.g500, lineHeight: 1.7 }}
        >
          A worker will verify on-site.
          <br />
          Green Coins are credited after collection.
        </div>
        <div
          style={{
            background: COLORS.cGreenL,
            borderRadius: 14,
            padding: 20,
            width: "100%",
          }}
        >
          <div style={{ fontSize: 12, color: COLORS.cGreen, marginBottom: 4 }}>
            {pred ? "AI classified as" : "AI is still analyzing"}
          </div>
          <div
            style={{ fontSize: 28, fontWeight: 900, color: COLORS.cGreen }}
          >
            {pred?.category_name || "…"}
          </div>
          <div
            style={{
              fontSize: 13,
              color: COLORS.cGreen,
              marginTop: 6,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 5,
            }}
          >
            <Coins size={15} /> up to {ESTIMATED_REWARD[stream] || 10}{" "}
            coins on completion
          </div>
        </div>
        <Btn onClick={reset} size="lg">
          Submit Another
        </Btn>
        <Btn
          onClick={() => navigate(`/citizen/reports/${report.id}`)}
          variant="ghost"
          color={COLORS.g500}
        >
          View Report <ChevronRight size={15} />
        </Btn>
      </div>
    );
  }

  return (
    <div
      style={{
        fontFamily: font,
        maxWidth: 720,
        display: "flex",
        flexDirection: "column",
        gap: 14,
      }}
    >
      <div>
        <div style={{ fontSize: 18, fontWeight: 800, color: COLORS.g900 }}>
          Submit Waste Report
        </div>
        <div style={{ fontSize: 11, color: COLORS.g400, marginTop: 3 }}>
          YOLOv11n · ONNX Runtime · 34 waste classes
        </div>
      </div>

      <Stepper step={step} />

      {step === 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ display: "flex", gap: 10 }}>
            <button
              onClick={() => setCameraOpen(true)}
              style={photoCard}
              onMouseEnter={(e) => {
                e.currentTarget.style.boxShadow = `0 0 0 2px ${COLORS.cGreen}`;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.boxShadow =
                  "0 1px 4px rgba(0,0,0,0.06)";
              }}
            >
              <Camera size={34} color={COLORS.cGreen} />
              <div style={photoCardTitle}>Take a photo</div>
              <div style={photoCardDesc}>Use your camera to scan waste</div>
              <div style={photoCardCta}>Tap to scan</div>
            </button>
            <button
              onClick={() => fileRef.current?.click()}
              style={photoCard}
              onMouseEnter={(e) => {
                e.currentTarget.style.boxShadow = `0 0 0 2px ${COLORS.cGreen}`;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.boxShadow =
                  "0 1px 4px rgba(0,0,0,0.06)";
              }}
            >
              <ImagePlus size={34} color={COLORS.dBlue} />
              <div style={photoCardTitle}>Upload a photo</div>
              <div style={photoCardDesc}>JPEG or PNG from your device</div>
              <div style={photoCardCta}>Tap to upload</div>
            </button>
          </div>
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
          {cameraOpen && (
            <CameraModal
              onCapture={(file) => {
                setCameraOpen(false);
                handleFile(file);
              }}
              onClose={() => setCameraOpen(false)}
            />
          )}
        </div>
      )}

      {step >= 1 && (
        <Card style={{ padding: 16 }}>
          <div
            style={{
              fontSize: 14,
              fontWeight: 700,
              color: COLORS.g900,
              marginBottom: 12,
            }}
          >
            {step === 1 ? "Review Photo" : "Analyzing…"}
          </div>
          {preview && (
            <img
              src={preview}
              alt="waste"
              style={{
                width: "100%",
                borderRadius: 10,
                maxHeight: 260,
                objectFit: "cover",
                marginBottom: 12,
              }}
            />
          )}
          {step === 1 && (
            <>
              <label
                style={{
                  fontSize: 12,
                  fontWeight: 600,
                  color: COLORS.g600,
                  display: "block",
                  marginBottom: 6,
                }}
              >
                Describe the waste
              </label>
              <textarea
                placeholder="e.g. pile of plastic bottles near the gate…"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={3}
                style={{
                  width: "100%",
                  boxSizing: "border-box",
                  border: `1px solid ${COLORS.g300}`,
                  borderRadius: 8,
                  padding: "10px 12px",
                  fontSize: 14,
                  fontFamily: font,
                  marginBottom: 12,
                }}
              />
            </>
          )}
          {error && (
            <div
              style={{
                background: COLORS.redL,
                borderRadius: 8,
                padding: 10,
                fontSize: 12,
                color: COLORS.red,
                marginBottom: 10,
                display: "flex",
                gap: 6,
                alignItems: "center",
              }}
            >
              <TriangleAlert size={15} /> {error}
            </div>
          )}
          {step === 2 && (
            <div
              style={{
                marginBottom: 12,
                display: "flex",
                flexDirection: "column",
                gap: 7,
              }}
            >
              {[
                "Uploading image",
                "YOLOv11n detection (ONNX)",
                "Waste-stream mapping",
                "Saving AI result",
              ].map((s) => (
                <div
                  key={s}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    fontSize: 12,
                    color: COLORS.g400,
                  }}
                >
                  <Spinner color={COLORS.dBlue} />
                  {s}
                </div>
              ))}
            </div>
          )}
          <div style={{ display: "flex", gap: 10 }}>
            {step === 1 && (
              <>
                <Btn onClick={reset} variant="ghost" color={COLORS.g500}>
                  <ArrowLeft size={0} /> Back
                </Btn>
                <Btn
                  onClick={handleSubmit}
                  style={{ flex: 1, justifyContent: "center" }}
                  disabled={submitting || !imageFile}
                >
                  {submitting ? (
                    <>
                      <Spinner /> Submitting…
                    </>
                  ) : (
                    "Analyze & Submit"
                  )}
                </Btn>
              </>
            )}
          </div>
        </Card>
      )}

      {report?.prediction && step !== 3 && (
        <AnalysisResult report={report} />
      )}
    </div>
  );
}

function Stepper({ step }: { step: number }) {
  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
      {STEPS.map((s, i) => (
        <div
          key={s}
          style={{ display: "flex", alignItems: "center", flex: i < 3 ? 1 : "none" }}
        >
          <div
            style={{
              width: 22,
              height: 22,
              borderRadius: "50%",
              background:
                step > i
                  ? COLORS.cGreen
                  : step === i
                    ? COLORS.dBlue
                    : COLORS.g200,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#fff",
              fontSize: 10,
              fontWeight: 700,
              flexShrink: 0,
            }}
          >
            {step > i ? <Check size={12} /> : i + 1}
          </div>
          <div
            style={{
              fontSize: 10,
              color:
                step === i
                  ? COLORS.dBlue
                  : step > i
                    ? COLORS.cGreen
                    : COLORS.g400,
              margin: "0 3px",
              whiteSpace: "nowrap",
            }}
          >
            {s}
          </div>
          {i < 3 && (
            <div
              style={{
                flex: 1,
                height: 2,
                background: step > i ? COLORS.cGreen : COLORS.g200,
                borderRadius: 1,
              }}
            />
          )}
        </div>
      ))}
    </div>
  );
}

/** Real backend prediction rendered in the artifact's result style. */
export function AnalysisResult({ report }: { report: Report }) {
  const pred = report.prediction;
  if (!pred) return null;
  const stream = toStreamKey(pred.category_name);
  const conf = Number(pred.confidence ?? 0);
  const unclassified = pred.model_used === "yolo_no_detection";
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <Card
        style={{
          padding: 16,
          borderLeft: `4px solid ${unclassified ? COLORS.g300 : COLORS.cGreen}`,
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            marginBottom: 12,
          }}
        >
          <div>
            <div
              style={{
                fontSize: 10,
                color: COLORS.g400,
                textTransform: "uppercase",
                letterSpacing: 0.5,
                marginBottom: 4,
              }}
            >
              AI Classification
            </div>
            <div style={{ fontSize: 22, fontWeight: 900, color: COLORS.g900 }}>
              {unclassified ? "Unclassified" : pred.category_name}
            </div>
            {!unclassified && <WastePill category={pred.category_name} />}
          </div>
          <div style={{ textAlign: "right" }}>
            <div
              style={{ fontSize: 28, fontWeight: 900, color: COLORS.cGreen }}
            >
              {unclassified ? "—" : `${conf.toFixed(0)}%`}
            </div>
            <div style={{ fontSize: 10, color: COLORS.g400 }}>
              Model confidence
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <Bar v={conf} max={100} color={COLORS.cGreen} />
          <span style={{ fontSize: 12, fontWeight: 700, color: COLORS.cGreen }}>
            {pred.model_used || "spiro_pipeline"}
          </span>
        </div>
        {unclassified && (
          <div style={{ fontSize: 12, color: COLORS.g500, fontStyle: "italic" }}>
            The AI couldn&apos;t identify the waste — a worker will verify
            the category during collection.
          </div>
        )}
      </Card>
      {!unclassified && (
        <Card style={{ padding: 16 }}>
          <div
            style={{
              fontSize: 14,
              fontWeight: 700,
              color: COLORS.g900,
              marginBottom: 10,
            }}
          >
            What to do
          </div>
          {DISPOSAL_GUIDE[stream].map((inst, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                gap: 8,
                alignItems: "center",
                padding: "5px 0",
                fontSize: 13,
                color: COLORS.g700,
              }}
            >
              <span style={{ color: COLORS.cGreen, fontWeight: 700 }}>✓</span>
              {inst}
            </div>
          ))}
          <div
            style={{
              marginTop: 10,
              padding: 10,
              background: COLORS.redL,
              borderRadius: 8,
            }}
          >
            <div
              style={{
                fontSize: 11,
                fontWeight: 700,
                color: COLORS.red,
                marginBottom: 4,
              }}
            >
              Do not mix with:
            </div>
            {DO_NOT_MIX[stream].map((d, i) => (
              <div key={i} style={{ fontSize: 12, color: COLORS.red }}>
                • {d}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

/** Fullscreen camera capture (artifact's CameraCapture, wired to submit). */
function CameraModal({
  onCapture,
  onClose,
}: {
  onCapture: (file: File) => void;
  onClose: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffectOnce(() => {
    let s: MediaStream | null = null;
    let cancelled = false;
    async function start() {
      try {
        s = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "environment",
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
        });
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        setStream(s);
        if (videoRef.current) {
          videoRef.current.srcObject = s;
          videoRef.current.onloadedmetadata = () => setReady(true);
        }
      } catch (err: any) {
        if (!cancelled)
          setError(
            err?.name === "NotAllowedError"
              ? "Camera permission denied. Allow camera access, or upload a photo instead."
              : `Camera error: ${err?.message || "unknown"}. Upload a photo instead.`
          );
      }
    }
    start();
    return () => {
      cancelled = true;
      if (s) s.getTracks().forEach((t) => t.stop());
    };
  });

  function capture() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    canvas.toBlob(
      (blob) => {
        if (blob) {
          onCapture(
            new File([blob], `capture-${Date.now()}.jpg`, {
              type: "image/jpeg",
            })
          );
        }
      },
      "image/jpeg",
      0.92
    );
    stream?.getTracks().forEach((t) => t.stop());
  }

  function close() {
    stream?.getTracks().forEach((t) => t.stop());
    onClose();
  }

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "#000",
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div
        style={{
          padding: "12px 16px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexShrink: 0,
        }}
      >
        <span style={{ color: "#fff", fontSize: 16, fontWeight: 700 }}>
          <Camera size={16} /> Scan Waste
        </span>
        <button
          onClick={close}
          style={{
            background: "rgba(255,255,255,.2)",
            border: "none",
            color: "#fff",
            borderRadius: 8,
            padding: "6px 14px",
            cursor: "pointer",
            fontSize: 14,
            fontWeight: 600,
          }}
        >
          ✕ Close
        </button>
      </div>
      {error ? (
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            padding: 24,
            gap: 16,
          }}
        >
          <Camera size={48} color="#fff" />
          <div
            style={{
              color: "#fff",
              fontSize: 14,
              textAlign: "center",
              lineHeight: 1.6,
            }}
          >
            {error}
          </div>
          <button
            onClick={close}
            style={{
              background: COLORS.cGreen,
              color: "#fff",
              border: "none",
              borderRadius: 10,
              padding: "10px 24px",
              fontSize: 14,
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            Upload Photo Instead
          </button>
        </div>
      ) : (
        <>
          <div
            style={{ flex: 1, position: "relative", overflow: "hidden" }}
          >
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ width: "100%", height: "100%", objectFit: "cover" }}
            />
            <div
              style={{
                position: "absolute",
                inset: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                pointerEvents: "none",
              }}
            >
              <div style={{ width: 260, height: 260, position: "relative" }}>
                {(
                  [
                    ["t", "l"],
                    ["t", "r"],
                    ["b", "l"],
                    ["b", "r"],
                  ] as const
                ).map(([v, h]) => (
                  <div
                    key={v + h}
                    style={{
                      position: "absolute",
                      width: 32,
                      height: 32,
                      top: v === "t" ? 0 : "auto",
                      bottom: v === "b" ? 0 : "auto",
                      left: h === "l" ? 0 : "auto",
                      right: h === "r" ? 0 : "auto",
                      borderTop:
                        v === "t" ? `3px solid ${COLORS.cGreen}` : "none",
                      borderBottom:
                        v === "b" ? `3px solid ${COLORS.cGreen}` : "none",
                      borderLeft:
                        h === "l" ? `3px solid ${COLORS.cGreen}` : "none",
                      borderRight:
                        h === "r" ? `3px solid ${COLORS.cGreen}` : "none",
                    }}
                  />
                ))}
              </div>
            </div>
            <div
              style={{
                position: "absolute",
                bottom: 20,
                left: 0,
                right: 0,
                textAlign: "center",
                color: "rgba(255,255,255,.7)",
                fontSize: 12,
              }}
            >
              Point at waste and tap capture
            </div>
          </div>
          <div
            style={{
              padding: 20,
              display: "flex",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <button
              onClick={capture}
              disabled={!ready}
              style={{
                width: 72,
                height: 72,
                borderRadius: "50%",
                background: ready ? COLORS.cGreen : "#555",
                border: "4px solid #fff",
                cursor: ready ? "pointer" : "not-allowed",
                color: "#fff",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Camera size={28} />
            </button>
          </div>
        </>
      )}
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
}

function useEffectOnce(fn: () => void | (() => void)) {
  useEffect(fn, []);
}

const photoCard: React.CSSProperties = {
  flex: 1,
  cursor: "pointer",
  borderRadius: 14,
  overflow: "hidden",
  border: `2px solid ${COLORS.g200}`,
  background: COLORS.white,
  boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
  transition: "box-shadow .15s",
  padding: "22px 12px",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  gap: 4,
};

const photoCardTitle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 700,
  color: COLORS.g900,
};

const photoCardDesc: React.CSSProperties = {
  fontSize: 11,
  color: COLORS.g500,
  lineHeight: 1.4,
};

const photoCardCta: React.CSSProperties = {
  background: COLORS.cGreen,
  borderRadius: 8,
  padding: "5px 0",
  textAlign: "center",
  fontSize: 12,
  fontWeight: 700,
  color: "#fff",
  width: "100%",
  marginTop: 6,
};

export default SubmitReport;
