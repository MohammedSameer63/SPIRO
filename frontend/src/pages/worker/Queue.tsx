import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Check, MapPin } from "lucide-react";
import {
  getWorkerQueue,
  getWorkerWards,
  updateWorkerReportStatus,
} from "../../api/reports";
import { resolveImageUrl } from "../../api/config";
import type { Report } from "../../api/reports";
import {
  Btn,
  Card,
  EmptyState,
  PageHeader,
  StatusPill,
  WastePill,
} from "../../components/ui";
import { COLORS, font } from "../../theme";

function Queue() {
  const navigate = useNavigate();
  const [reports, setReports] = useState<Report[]>([]);
  const [hasWards, setHasWards] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [acceptingId, setAcceptingId] = useState<string | null>(null);

  useEffect(() => {
    loadReports();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadReports() {
    try {
      setLoading(true);
      setError("");
      const [queueResponse, wardsResponse] = await Promise.all([
        getWorkerQueue(),
        getWorkerWards().catch(() => null),
      ]);
      if (queueResponse.success) setReports(queueResponse.data);
      if (wardsResponse !== null) {
        setHasWards(
          wardsResponse.success && wardsResponse.data.length > 0
        );
      }
    } catch (err: any) {
      setError(err?.message || "Failed to load pending reports.");
    } finally {
      setLoading(false);
    }
  }

  async function handleAccept(reportId: string) {
    try {
      setAcceptingId(reportId);
      setError("");
      await updateWorkerReportStatus(reportId, "ACCEPTED");
      setReports((cur) => cur.filter((r) => r.id !== reportId));
    } catch (err: any) {
      setError(err?.message || "Failed to accept report.");
    } finally {
      setAcceptingId(null);
    }
  }

  return (
    <div style={{ fontFamily: font, maxWidth: 1120 }}>
      <PageHeader
        title="Today's Assignments"
        subtitle={`${reports.length} pending in your wards`}
        right={
          <Btn size="sm" variant="ghost" onClick={loadReports}>
            Refresh
          </Btn>
        }
      />

      {loading && <p style={{ color: COLORS.g500 }}>Loading queue…</p>}

      {!loading && error && (
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

      {!loading && !error && reports.length === 0 && (
        <EmptyState
          icon={<Check size={40} />}
          title={hasWards ? "All caught up!" : "No wards assigned"}
          hint={
            hasWards
              ? "No pending reports in your wards."
              : "You are not assigned to any ward yet — contact your admin."
          }
          action={<Btn onClick={loadReports}>Refresh Queue</Btn>}
        />
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))",
          gap: 12,
        }}
      >
        {reports.map((report, i) => (
          <Card key={report.id} style={{ padding: 0, overflow: "hidden" }}>
            <div style={{ display: "flex" }}>
              <div
                style={{
                  width: 44,
                  background: COLORS.dBlue,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#fff",
                  fontSize: 16,
                  fontWeight: 800,
                  flexShrink: 0,
                }}
              >
                {i + 1}
              </div>
              <div style={{ flex: 1, padding: 14 }}>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "flex-start",
                    gap: 8,
                    marginBottom: 8,
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
                    <div
                      style={{
                        fontSize: 11,
                        color: COLORS.g500,
                        marginTop: 2,
                        display: "flex",
                        alignItems: "center",
                        gap: 4,
                      }}
                    >
                      <MapPin size={11} />
                      {report.address || "No address"}
                    </div>
                  </div>
                  <StatusPill status={report.status} />
                </div>
                {report.image_url && (
                  <img
                    src={resolveImageUrl(report.image_url)}
                    alt="Reported waste"
                    style={{
                      width: "100%",
                      maxHeight: 180,
                      objectFit: "cover",
                      borderRadius: 10,
                      marginBottom: 10,
                    }}
                  />
                )}
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    gap: 8,
                  }}
                >
                  <div style={{ fontSize: 11, color: COLORS.g500 }}>
                    {report.prediction ? (
                      <WastePill
                        category={report.prediction.category_name}
                      />
                    ) : (
                      "AI analyzing…"
                    )}
                    <div style={{ marginTop: 3 }}>
                      Reported{" "}
                      {new Date(report.created_at).toLocaleDateString()}
                    </div>
                  </div>
                  <Btn
                    onClick={() => handleAccept(report.id)}
                    disabled={acceptingId === report.id}
                    size="sm"
                  >
                    {acceptingId === report.id ? "Accepting…" : "Accept"}
                  </Btn>
                </div>
                <button
                  onClick={() => navigate(`/worker/active`)}
                  style={{
                    border: "none",
                    background: "none",
                    color: COLORS.dBlue,
                    fontSize: 11,
                    fontWeight: 600,
                    cursor: "pointer",
                    padding: 0,
                    marginTop: 8,
                  }}
                >
                  Accept, then verify pickup in Active Reports →
                </button>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

export default Queue;
