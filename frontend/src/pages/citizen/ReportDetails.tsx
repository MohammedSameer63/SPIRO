import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Coins, MapPin, RefreshCw } from "lucide-react";
import {
  getReport,
  getReportPrediction,
  retryReportAnalysis,
  type Prediction,
  type Report,
} from "../../api/reports";
import { resolveImageUrl } from "../../api/config";
import { getWallet } from "../../api/coins";
import { AnalysisResult } from "./SubmitReport";
import { ErrorBoundary } from "../../components/ErrorBoundary";
import {
  Btn,
  Card,
  EmptyState,
  PageHeader,
  StatusPill,
} from "../../components/ui";
import { COLORS, font } from "../../theme";

function ReportDetails() {
  const { id } = useParams();

  const [report, setReport] = useState<Report | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [retrying, setRetrying] = useState(false);
  const [coinsEarned, setCoinsEarned] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    loadReport(id, cancelled);
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  async function loadReport(reportId: string, cancelled = false) {
    try {
      setLoading(true);
      setPredictionLoading(true);
      setError("");
      const response = await getReport(reportId);
      if (cancelled) return;
      setReport(response);

      try {
        const wallet = await getWallet();
        if (cancelled) return;
        const earned = wallet.transactions
          .filter((tx) => tx.report_id === reportId)
          .reduce((sum, tx) => sum + (Number(tx.amount) || 0), 0);
        setCoinsEarned(earned > 0 ? earned : null);
      } catch {
        if (!cancelled) setCoinsEarned(null);
      }

      if (response.prediction) {
        setPrediction(response.prediction);
        setPredictionLoading(false);
      } else {
        try {
          const pred = await getReportPrediction(reportId);
          if (cancelled) return;
          setPrediction(pred);
        } catch {
          if (cancelled) return;
          setPrediction(null);
        } finally {
          if (!cancelled) setPredictionLoading(false);
        }
      }
    } catch (err: any) {
      if (cancelled) return;
      setError(err?.message || "Failed to load report.");
      setPredictionLoading(false);
    } finally {
      if (!cancelled) setLoading(false);
    }
  }

  async function handleRetryAnalysis() {
    if (!id || retrying) return;
    try {
      setRetrying(true);
      setError("");
      const updated = await retryReportAnalysis(id);
      setReport(updated);
      setPrediction(updated.prediction ?? null);
    } catch (err: any) {
      setError(err?.message || "AI analysis failed. Please try again.");
    } finally {
      setRetrying(false);
    }
  }

  if (loading) {
    return (
      <div style={{ fontFamily: font }}>
        <PageHeader title="Report Details" />
        <p style={{ color: COLORS.g500 }}>Loading report…</p>
      </div>
    );
  }

  if (error && !report) {
    return (
      <div style={{ fontFamily: font }}>
        <Link to="/citizen/reports" style={backStyle}>
          <ArrowLeft size={14} /> Back to My Reports
        </Link>
        <EmptyState
          icon={<MapPin size={40} />}
          title="Couldn't load report"
          hint={error}
        />
      </div>
    );
  }

  if (!report) return null;
  const withPrediction: Report = { ...report, prediction };

  return (
    <div style={{ fontFamily: font, maxWidth: 860 }}>
      <Link to="/citizen/reports" style={backStyle}>
        <ArrowLeft size={14} /> Back to My Reports
      </Link>
      <PageHeader
        title="Report Details"
        subtitle={`Filed ${new Date(report.created_at).toLocaleString()}`}
        right={<StatusPill status={report.status} />}
      />

      {error && (
        <Card
          style={{
            padding: 12,
            marginBottom: 12,
            background: COLORS.redL,
            fontSize: 13,
            color: COLORS.red,
          }}
        >
          {error}
        </Card>
      )}

      {report.image_url && (
        <img
          src={resolveImageUrl(report.image_url)}
          alt="Reported waste"
          style={{
            width: "100%",
            maxHeight: 420,
            objectFit: "cover",
            borderRadius: 16,
            border: `1px solid ${COLORS.g200}`,
            marginBottom: 14,
          }}
          onError={(e) => {
            e.currentTarget.style.display = "none";
          }}
        />
      )}

      <Card style={{ padding: 16, marginBottom: 14 }}>
        <div
          style={{
            fontSize: 14,
            fontWeight: 700,
            color: COLORS.g900,
            marginBottom: 10,
          }}
        >
          Waste Information
        </div>
        {[
          ["Description", report.description || "No description"],
          ["Address", report.address || "No address"],
          [
            "Location",
            report.latitude != null
              ? `${report.latitude}, ${report.longitude}`
              : "Not pinned",
          ],
          ["Status", report.status],
        ].map(([k, v]) => (
          <div
            key={k}
            style={{
              display: "flex",
              justifyContent: "space-between",
              gap: 12,
              padding: "7px 0",
              borderBottom: `1px solid ${COLORS.g100}`,
              fontSize: 13,
            }}
          >
            <span style={{ color: COLORS.g500 }}>{k}</span>
            <span style={{ fontWeight: 600, color: COLORS.g900, textAlign: "right" }}>
              {v}
            </span>
          </div>
        ))}
        {coinsEarned !== null && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              marginTop: 12,
              background: COLORS.cGreenL,
              borderRadius: 10,
              padding: "10px 12px",
              fontSize: 14,
              fontWeight: 700,
              color: COLORS.cGreenD,
            }}
          >
            <Coins size={16} /> +{coinsEarned} Green Coins earned
          </div>
        )}
      </Card>

      <ErrorBoundary backTo="/citizen/reports">
        <AnalysisResult report={withPrediction} />
        {!predictionLoading && !prediction && (
          <Btn
            onClick={handleRetryAnalysis}
            disabled={retrying}
            style={{ marginTop: 12 }}
          >
            <RefreshCw size={15} />
            {retrying ? "Analyzing…" : "Retry AI Analysis"}
          </Btn>
        )}
        {predictionLoading && (
          <p style={{ fontSize: 12, color: COLORS.g400 }}>
            Fetching AI result…
          </p>
        )}
      </ErrorBoundary>
    </div>
  );
}

const backStyle: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 5,
  marginBottom: 16,
  color: COLORS.cGreen,
  textDecoration: "none",
  fontSize: 13,
  fontWeight: 600,
};

export default ReportDetails;
