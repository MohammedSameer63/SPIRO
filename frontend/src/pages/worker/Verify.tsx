import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  ArrowLeft,
  Check,
  MapPin,
  Navigation,
  QrCode,
  ShieldCheck,
  TriangleAlert,
} from "lucide-react";
import {
  getWorkerActiveReports,
  updateWorkerReportStatus,
  type Report,
} from "../../api/reports";
import { validatePickup } from "../../api/qr";
import { QrScanner } from "../../components/QrScanner";
import {
  Btn,
  Card,
  EmptyState,
  PageHeader,
  Spinner,
  StatusPill,
  WastePill,
} from "../../components/ui";
import { COLORS, font } from "../../theme";

type Phase = "gps" | "scan" | "result";

type Geo = { lat: number; lng: number; accuracy: number } | null;

function Verify() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [phase, setPhase] = useState<Phase>("gps");
  const [geo, setGeo] = useState<Geo>(null);
  const [geoError, setGeoError] = useState("");
  const [locating, setLocating] = useState(false);

  const [validating, setValidating] = useState(false);
  const [qrOk, setQrOk] = useState(false);
  const [decided, setDecided] = useState(false);
  const [acting, setActing] = useState(false);

  useEffect(() => {
    if (id) loadReport(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  async function loadReport(reportId: string) {
    try {
      setLoading(true);
      setError("");
      const res = await getWorkerActiveReports();
      const found = (res.success ? res.data : []).find(
        (r) => r.id === reportId
      );
      if (!found) {
        throw new Error(
          "Report is not in your active list. Accept it from the queue first."
        );
      }
      setReport(found);
      if (found.pickup_validated) {
        setQrOk(true);
        setPhase("result");
      }
    } catch (err: any) {
      setError(err?.message || "Failed to load report.");
    } finally {
      setLoading(false);
    }
  }

  function captureGPS() {
    if (!navigator.geolocation) {
      setGeoError("Geolocation is not supported on this device.");
      return;
    }
    setLocating(true);
    setGeoError("");
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setGeo({
          lat: pos.coords.latitude,
          lng: pos.coords.longitude,
          accuracy: pos.coords.accuracy,
        });
        setLocating(false);
      },
      () => {
        setGeoError(
          "Location permission denied. You can still continue to the QR scan."
        );
        setLocating(false);
      },
      { enableHighAccuracy: true, timeout: 15000 }
    );
  }

  async function handleScan(payload: string) {
    if (!id || validating) return;
    try {
      setValidating(true);
      setError("");
      await validatePickup(id, payload);
      setQrOk(true);
      setPhase("result");
      setReport((r) => (r ? { ...r, pickup_validated: true } : r));
    } catch (err: any) {
      setError(err?.message || "QR validation failed.");
    } finally {
      setValidating(false);
    }
  }

  async function handleStatus(status: Report["status"]) {
    if (!id || acting) return;
    try {
      setActing(true);
      setError("");
      await updateWorkerReportStatus(id, status);
      if (status === "COMPLETED") {
        setDecided(true);
      } else {
        setReport((r) => (r ? { ...r, status } : r));
      }
    } catch (err: any) {
      setError(err?.message || "Failed to update report.");
    } finally {
      setActing(false);
    }
  }

  const distance = distanceToReport(geo, report);

  return (
    <div
      style={{ fontFamily: font, maxWidth: 720, display: "flex", flexDirection: "column", gap: 14 }}
    >
      <button onClick={() => navigate("/worker/active")} style={backBtn}>
        <ArrowLeft size={14} /> Back to Active
      </button>
      <PageHeader title="Pickup Verification" subtitle="GPS → QR scan → confirm" />

      {loading && <p style={{ color: COLORS.g500 }}>Loading report…</p>}
      {!loading && error && !report && (
        <EmptyState
          icon={<QrCode size={40} />}
          title="Can't verify this report"
          hint={error}
          action={<Btn onClick={() => navigate("/worker/active")}>Back to Active</Btn>}
        />
      )}

      {report && (
        <>
          <Card
            style={{
              padding: 14,
              borderLeft: `4px solid ${COLORS.dBlue}`,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 15, fontWeight: 700, color: COLORS.g900 }}>
                  {report.description || "Waste report"}
                </div>
                <div
                  style={{
                    fontSize: 12,
                    color: COLORS.g500,
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                    marginTop: 2,
                  }}
                >
                  <MapPin size={12} /> {report.address || "No address"}
                </div>
              </div>
              <StatusPill status={report.status} />
            </div>
            <div style={{ marginTop: 8 }}>
              {report.prediction ? (
                <WastePill category={report.prediction.category_name} />
              ) : (
                <span style={{ fontSize: 11, color: COLORS.g400 }}>AI analyzing…</span>
              )}
            </div>
          </Card>

          <Stepper phase={phase} decided={decided} />

          {error && (
            <Card
              style={{ padding: 12, background: COLORS.redL, fontSize: 13, color: COLORS.red }}
            >
              {error}
            </Card>
          )}

          {/* Phase 1 — GPS */}
          {(phase === "gps" || (!geo && phase !== "result")) && !decided && (
            <Card style={{ padding: 16 }}>
              <StepTitle n="1" title="Capture Location" />
              {geo ? (
                <div
                  style={{
                    background: COLORS.cGreenL,
                    borderRadius: 8,
                    padding: 10,
                    fontSize: 12,
                    color: COLORS.cGreen,
                    fontWeight: 600,
                  }}
                >
                  <Navigation size={13} /> GPS acquired · {geo.lat.toFixed(4)}°,
                  {" "}{geo.lng.toFixed(4)}° · ±{Math.round(geo.accuracy)}m
                  {distance != null && (
                    <div style={{ marginTop: 4 }}>
                      ≈{distance}m from the reported location (informational)
                    </div>
                  )}
                </div>
              ) : (
                <>
                  <Btn
                    onClick={captureGPS}
                    disabled={locating}
                    style={{ width: "100%", justifyContent: "center" }}
                  >
                    {locating ? (
                      <>
                        <Spinner /> Locating…
                      </>
                    ) : (
                      "Capture Location"
                    )}
                  </Btn>
                  {geoError && (
                    <div style={{ fontSize: 12, color: COLORS.amber, marginTop: 8 }}>
                      <TriangleAlert size={13} /> {geoError}
                    </div>
                  )}
                </>
              )}
              {(geo || geoError) && phase === "gps" && (
                <Btn
                  onClick={() => setPhase("scan")}
                  color={COLORS.dBlue}
                  style={{ width: "100%", justifyContent: "center", marginTop: 12 }}
                  size="lg"
                >
                  Proceed to QR Scan →
                </Btn>
              )}
            </Card>
          )}

          {/* Phase 2 — QR scan */}
          {phase === "scan" && !decided && (
            <Card style={{ padding: 16 }}>
              <StepTitle n="2" title="Scan Household QR" />
              <div style={{ fontSize: 12, color: COLORS.g500, marginBottom: 4 }}>
                HMAC-signed · verified server-side against the report&apos;s
                household
              </div>
              {validating ? (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    fontSize: 13,
                    color: COLORS.g600,
                    padding: "12px 0",
                  }}
                >
                  <Spinner color={COLORS.dBlue} /> Validating scan…
                </div>
              ) : (
                <QrScanner onScan={handleScan} />
              )}
            </Card>
          )}

          {/* Phase 3 — result + confirm */}
          {phase === "result" && qrOk && !decided && (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <Card
                style={{
                  padding: 14,
                  background: COLORS.cGreenL,
                  borderLeft: `4px solid ${COLORS.cGreen}`,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <ShieldCheck size={28} color={COLORS.cGreen} />
                  <div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.g900 }}>
                      QR Validated{geo ? " · GPS Captured" : ""}
                    </div>
                    <div style={{ fontSize: 12, color: COLORS.g600 }}>
                      Pickup verified — the citizen earns bonus Green Coins on
                      completion.
                    </div>
                  </div>
                </div>
              </Card>
              {report.status === "ACCEPTED" && (
                <Btn
                  onClick={() => handleStatus("IN_PROGRESS")}
                  disabled={acting}
                  style={{ width: "100%", justifyContent: "center" }}
                  size="lg"
                >
                  {acting ? "Updating…" : "Start Work"}
                </Btn>
              )}
              {report.status === "IN_PROGRESS" && (
                <Btn
                  onClick={() => handleStatus("COMPLETED")}
                  disabled={acting}
                  style={{ width: "100%", justifyContent: "center" }}
                  size="lg"
                >
                  {acting ? (
                    "Completing…"
                  ) : (
                    <>
                      <Check size={16} /> Confirm Collection
                    </>
                  )}
                </Btn>
              )}
            </div>
          )}

          {decided && (
            <Card style={{ padding: 32, textAlign: "center" }}>
              <Check size={56} color={COLORS.cGreen} />
              <div
                style={{ fontSize: 18, fontWeight: 800, color: COLORS.g900, marginTop: 8 }}
              >
                Collection Confirmed!
              </div>
              <div style={{ fontSize: 13, color: COLORS.g500, marginTop: 6 }}>
                Green Coins credited to the citizen wallet.
              </div>
              <Btn
                onClick={() => navigate("/worker/active")}
                style={{ marginTop: 14 }}
              >
                Back to Active Reports
              </Btn>
            </Card>
          )}
        </>
      )}
    </div>
  );
}

function Stepper({ phase, decided }: { phase: Phase; decided: boolean }) {
  const steps = ["GPS", "QR Scan", "Confirm"];
  const idx = phase === "gps" ? 0 : phase === "scan" ? 1 : 2;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
      {steps.map((s, i) => {
        const done = decided || i < idx || (i === 2 && decided);
        const active = !decided && i === idx;
        return (
          <div key={s} style={{ display: "flex", alignItems: "center", flex: i < 2 ? 1 : "none" }}>
            <div
              style={{
                width: 26,
                height: 26,
                borderRadius: "50%",
                background: done ? COLORS.cGreen : active ? COLORS.dBlue : COLORS.g200,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#fff",
                fontSize: 11,
                fontWeight: 700,
                flexShrink: 0,
              }}
            >
              {done ? <Check size={13} /> : i + 1}
            </div>
            <div
              style={{
                fontSize: 10,
                color: active ? COLORS.dBlue : done ? COLORS.cGreen : COLORS.g400,
                fontWeight: active || done ? 700 : 400,
                margin: "0 4px",
                whiteSpace: "nowrap",
              }}
            >
              {s}
            </div>
            {i < 2 && (
              <div style={{ flex: 1, height: 2, background: done ? COLORS.cGreen : COLORS.g200, borderRadius: 1 }} />
            )}
          </div>
        );
      })}
    </div>
  );
}

function StepTitle({ n, title }: { n: string; title: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
      <div
        style={{
          width: 26,
          height: 26,
          borderRadius: "50%",
          background: COLORS.dBlue,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#fff",
          fontSize: 11,
          fontWeight: 700,
        }}
      >
        {n}
      </div>
      <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.g900 }}>{title}</div>
    </div>
  );
}

/** Haversine distance in meters when both points exist (informational). */
function distanceToReport(geo: Geo, report: Report | null): number | null {
  if (!geo || !report || report.latitude == null || report.longitude == null)
    return null;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const R = 6371000;
  const dLat = toRad(Number(report.latitude) - geo.lat);
  const dLng = toRad(Number(report.longitude) - geo.lng);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(geo.lat)) *
      Math.cos(toRad(Number(report.latitude))) *
      Math.sin(dLng / 2) ** 2;
  return Math.round(2 * R * Math.asin(Math.sqrt(a)));
}

const backBtn: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 5,
  border: "none",
  background: "none",
  color: COLORS.g500,
  fontSize: 13,
  fontWeight: 600,
  cursor: "pointer",
  padding: 0,
  alignSelf: "flex-start",
};

export default Verify;
