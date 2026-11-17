import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { MapPin, ScanLine, Truck } from "lucide-react";
import {
  getWorkerActiveReports,
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

function ActiveReports() {
  const navigate = useNavigate();
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actingId, setActingId] = useState<string | null>(null);

  useEffect(() => {
    loadReports();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadReports() {
    try {
      setLoading(true);
      setError("");
      const response = await getWorkerActiveReports();
      if (response.success) setReports(response.data);
    } catch (err: any) {
      setError(err?.message || "Failed to load active reports.");
    } finally {
      setLoading(false);
    }
  }

  async function transition(reportId: string, status: Report["status"]) {
    try {
      setActingId(reportId);
      setError("");
      await updateWorkerReportStatus(reportId, status);
      if (status === "COMPLETED") {
        setReports((cur) => cur.filter((r) => r.id !== reportId));
      } else {
        setReports((cur) =>
          cur.map((r) => (r.id === reportId ? { ...r, status } : r))
        );
      }
    } catch (err: any) {
      setError(err?.message || "Failed to update report.");
    } finally {
      setActingId(null);
    }
  }

  return (
    <div style={{ fontFamily: font, maxWidth: 1120 }}>
      <PageHeader
        title="Active Reports"
        subtitle="Accepted work — verify pickup, then complete"
        right={
          <Btn size="sm" variant="ghost" onClick={loadReports}>
            Refresh
          </Btn>
        }
      />

      {loading && <p style={{ color: COLORS.g500 }}>Loading…</p>}

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
          icon={<Truck size={40} />}
          title="No active reports"
          hint="Accept reports from the queue to get started."
          action={<Btn onClick={() => navigate("/worker/queue")}>Open Queue</Btn>}
        />
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))",
          gap: 12,
        }}
      >
        {reports.map((report) => (
          <Card key={report.id} style={{ padding: 14 }}>
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
                  style={{ fontSize: 13, fontWeight: 700, color: COLORS.g900 }}
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
                  maxHeight: 170,
                  objectFit: "cover",
                  borderRadius: 10,
                  marginBottom: 10,
                }}
              />
            )}

            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 10,
              }}
            >
              {report.prediction ? (
                <WastePill category={report.prediction.category_name} />
              ) : (
                <span style={{ fontSize: 11, color: COLORS.g400 }}>
                  AI analyzing…
                </span>
              )}
              {report.pickup_validated && (
                <span
                  style={{ fontSize: 11, fontWeight: 700, color: COLORS.cGreen }}
                >
                  ✓ QR-validated
                </span>
              )}
            </div>

            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {!report.pickup_validated && (
                <Btn
                  size="sm"
                  onClick={() => navigate(`/worker/verify/${report.id}`)}
                >
                  <ScanLine size={14} /> Verify Pickup
                </Btn>
              )}
              {report.status === "ACCEPTED" && (
                <Btn
                  size="sm"
                  variant="ghost"
                  disabled={actingId === report.id}
                  onClick={() => transition(report.id, "IN_PROGRESS")}
                >
                  Start Work
                </Btn>
              )}
              {report.status === "IN_PROGRESS" && (
                <Btn
                  size="sm"
                  variant="ghost"
                  disabled={actingId === report.id}
                  onClick={() => transition(report.id, "COMPLETED")}
                >
                  Mark Completed
                </Btn>
              )}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

export default ActiveReports;
