import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ClipboardList, Plus } from "lucide-react";
import { getMyReports, type Report } from "../../api/reports";
import {
  Btn,
  Card,
  CoinTag,
  EmptyState,
  PageHeader,
  StatusPill,
  WastePill,
} from "../../components/ui";
import { COLORS, font } from "../../theme";

const FILTERS = ["All", "PENDING", "ACCEPTED", "IN_PROGRESS", "COMPLETED"] as const;

function Reports() {
  const navigate = useNavigate();
  const [reports, setReports] = useState<Report[]>([]);
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>("All");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    loadReports();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadReports() {
    try {
      setLoading(true);
      setError("");
      setReports(await getMyReports());
    } catch (err: any) {
      setError(err?.message || "Failed to load reports.");
    } finally {
      setLoading(false);
    }
  }

  const shown =
    filter === "All" ? reports : reports.filter((r) => r.status === filter);

  return (
    <div style={{ fontFamily: font, maxWidth: 1120 }}>
      <PageHeader
        title="My Reports"
        subtitle="Every report you filed, with live AI + worker status"
        right={
          <Btn size="sm" onClick={() => navigate("/citizen/report")}>
            <Plus size={14} /> New Report
          </Btn>
        }
      />
      <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
        {FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            style={{
              border: "none",
              borderRadius: 999,
              padding: "6px 14px",
              fontSize: 12,
              fontWeight: 700,
              cursor: "pointer",
              background: filter === f ? COLORS.cGreen : COLORS.g100,
              color: filter === f ? "#fff" : COLORS.g600,
            }}
          >
            {f === "All" ? "All" : f.charAt(0) + f.slice(1).toLowerCase()}
            {f !== "All" &&
              ` (${reports.filter((r) => r.status === f).length})`}
          </button>
        ))}
      </div>

      {loading && <p style={{ color: COLORS.g500 }}>Loading reports…</p>}

      {!loading && error && (
        <EmptyState
          icon={<ClipboardList size={40} />}
          title="Couldn't load reports"
          hint={error}
          action={<Btn onClick={loadReports}>Retry</Btn>}
        />
      )}

      {!loading && !error && shown.length === 0 && (
        <EmptyState
          icon={<ClipboardList size={40} />}
          title={reports.length === 0 ? "No reports yet" : "Nothing here"}
          hint={
            reports.length === 0
              ? "Submit your first waste report to get started."
              : "No reports match this filter."
          }
          action={
            reports.length === 0 ? (
              <Btn onClick={() => navigate("/citizen/report")}>
                Submit Your First Report
              </Btn>
            ) : undefined
          }
        />
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))",
          gap: 12,
        }}
      >
        {shown.map((report) => (
          <Card
            key={report.id}
            onClick={() => navigate(`/citizen/reports/${report.id}`)}
            style={{
              padding: 14,
              borderLeft: `4px solid ${borderFor(report)}`,
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-start",
                marginBottom: 6,
              }}
            >
              <div>
                <div
                  style={{
                    fontSize: 13,
                    fontWeight: 700,
                    color: COLORS.g900,
                  }}
                >
                  {report.description || "Waste report"}
                </div>
                <div style={{ fontSize: 11, color: COLORS.g400, marginTop: 2 }}>
                  {new Date(report.created_at).toLocaleDateString()} ·{" "}
                  {report.address || "No address"}
                </div>
              </div>
              <StatusPill status={report.status} />
            </div>
            <div
              style={{ display: "flex", gap: 8, alignItems: "center" }}
            >
              {report.prediction ? (
                report.prediction.model_used === "yolo_no_detection" ? (
                  <span style={{ fontSize: 11, color: COLORS.g500 }}>
                    AI: Unclassified
                  </span>
                ) : (
                  <WastePill category={report.prediction.category_name} />
                )
              ) : (
                <span style={{ fontSize: 11, color: COLORS.g400 }}>
                  AI: analyzing…
                </span>
              )}
              {typeof report.coins_awarded === "number" &&
                report.coins_awarded > 0 && (
                  <span style={{ marginLeft: "auto" }}>
                    <CoinTag amount={report.coins_awarded} />
                  </span>
                )}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

function borderFor(report: Report): string {
  if (report.status === "COMPLETED") return COLORS.cGreen;
  if (report.status === "PENDING") return COLORS.amber;
  if (report.status === "ACCEPTED") return COLORS.dBlue;
  return COLORS.sky;
}

export default Reports;
