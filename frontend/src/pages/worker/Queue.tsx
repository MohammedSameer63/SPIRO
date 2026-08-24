import { useEffect, useState } from "react";
import {
  getWorkerQueue,
  updateWorkerReportStatus,
} from "../../api/reports";
import type { Report } from "../../api/reports";

function Queue() {
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [acceptingId, setAcceptingId] = useState<string | null>(
    null
  );

  useEffect(() => {
    loadReports();
  }, []);

  async function loadReports() {
    try {
      setLoading(true);
      setError("");

      const response = await getWorkerQueue();

      if (response.success) {
        setReports(response.data);
      }
    } catch (err: any) {
      console.error(err);

      setError(
        err?.message ||
          "Failed to load pending reports."
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleAccept(reportId: string) {
    try {
      setAcceptingId(reportId);
      setError("");

      await updateWorkerReportStatus(
        reportId,
        "ACCEPTED"
      );

      setReports((currentReports) =>
        currentReports.filter(
          (report) => report.id !== reportId
        )
      );

      alert("Report accepted successfully!");
    } catch (err: any) {
      console.error(err);

      setError(
        err?.message ||
          "Failed to accept report."
      );
    } finally {
      setAcceptingId(null);
    }
  }

  if (loading) {
    return (
      <div>
        <h1>Report Queue</h1>

        <p style={styles.subtitle}>
          Pending reports from citizens.
        </p>

        <div style={styles.loading}>
          Loading pending reports...
        </div>
      </div>
    );
  }

  return (
    <div>
      <h1>Report Queue</h1>

      <p style={styles.subtitle}>
        Pending reports from citizens.
      </p>

      {error && (
        <div style={styles.error}>
          <span>{error}</span>

          <button
            onClick={loadReports}
            style={styles.retry}
          >
            Retry
          </button>
        </div>
      )}

      <div style={styles.container}>
        {reports.length === 0 ? (
          <div style={styles.empty}>
            <h2>No pending reports</h2>

            <p>
              There are currently no reports waiting
              for acceptance.
            </p>

            <button
              onClick={loadReports}
              style={styles.refreshButton}
            >
              Refresh Queue
            </button>
          </div>
        ) : (
          reports.map((report) => (
            <div
              key={report.id}
              style={styles.card}
            >
              <div style={styles.header}>
                <h2>Waste Report</h2>

                <span style={styles.status}>
                  {report.status}
                </span>
              </div>

              {report.image_url && (
                <img
                  src={
                    report.image_url.startsWith("http")
                      ? report.image_url
                      : `http://127.0.0.1:8000/${report.image_url.replace(
                          /^\/+/,
                          ""
                        )}`
                  }
                  alt="Reported waste"
                  style={styles.image}
                />
              )}

              <div style={styles.info}>
                <p>
                  <strong>Description:</strong>{" "}
                  {report.description ||
                    "No description"}
                </p>

                <p>
                  <strong>Address:</strong>{" "}
                  {report.address ||
                    "No address"}
                </p>

                <p>
                  <strong>Location:</strong>{" "}
                  {report.latitude},{" "}
                  {report.longitude}
                </p>

                <p>
                  <strong>Reported:</strong>{" "}
                  {new Date(
                    report.created_at
                  ).toLocaleString()}
                </p>
              </div>

              <button
                onClick={() =>
                  handleAccept(report.id)
                }
                disabled={
                  acceptingId === report.id
                }
                style={{
                  ...styles.button,
                  opacity:
                    acceptingId === report.id
                      ? 0.6
                      : 1,
                  cursor:
                    acceptingId === report.id
                      ? "not-allowed"
                      : "pointer",
                }}
              >
                {acceptingId === report.id
                  ? "Accepting..."
                  : "Accept Report"}
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

const styles = {
  subtitle: {
    color: "#6b7280",
    marginBottom: "25px",
  },

  container: {
    maxWidth: "850px",
    display: "flex",
    flexDirection: "column" as const,
    gap: "20px",
  },

  card: {
    padding: "25px",
    background: "white",
    borderRadius: "12px",
    boxShadow:
      "0 4px 15px rgba(0,0,0,0.08)",
  },

  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "15px",
  },

  status: {
    padding: "6px 12px",
    borderRadius: "20px",
    background: "#fef3c7",
    color: "#92400e",
    fontWeight: "bold",
    fontSize: "13px",
  },

  image: {
    width: "100%",
    maxHeight: "350px",
    objectFit: "cover" as const,
    borderRadius: "10px",
    marginBottom: "20px",
  },

  info: {
    lineHeight: "1.6",
  },

  button: {
    marginTop: "10px",
    padding: "11px 20px",
    border: "none",
    borderRadius: "7px",
    background: "#15803d",
    color: "white",
    fontSize: "15px",
    fontWeight: "bold",
  },

  empty: {
    padding: "35px",
    background: "white",
    borderRadius: "10px",
    textAlign: "center" as const,
    color: "#6b7280",
  },

  refreshButton: {
    marginTop: "15px",
    padding: "10px 18px",
    border: "none",
    borderRadius: "7px",
    background: "#15803d",
    color: "white",
    cursor: "pointer",
  },

  loading: {
    padding: "30px",
    background: "white",
    borderRadius: "10px",
    color: "#6b7280",
  },

  error: {
    display: "flex",
    alignItems: "center",
    padding: "15px",
    marginBottom: "20px",
    background: "#fee2e2",
    color: "#b91c1c",
    borderRadius: "8px",
  },

  retry: {
    marginLeft: "15px",
    padding: "8px 14px",
    border: "none",
    borderRadius: "6px",
    background: "#15803d",
    color: "white",
    cursor: "pointer",
  },
};

export default Queue;