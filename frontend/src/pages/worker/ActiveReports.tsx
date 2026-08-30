import { useEffect, useState } from "react";
import {
  getWorkerActiveReports,
  updateWorkerReportStatus,
} from "../../api/reports";
import type { Report } from "../../api/reports";

function ActiveReports() {
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    loadReports();
  }, []);

  async function loadReports() {
    try {
      setLoading(true);
      setError("");

      const response =
        await getWorkerActiveReports();

      if (response.success) {
        setReports(response.data);
      }
    } catch (err: any) {
      console.error(err);

      setError(
        err?.message ||
          "Failed to load active reports."
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleStart(reportId: string) {
    try {
      await updateWorkerReportStatus(
        reportId,
        "IN_PROGRESS"
      );

      setReports((currentReports) =>
        currentReports.map((report) =>
          report.id === reportId
            ? {
                ...report,
                status: "IN_PROGRESS",
              }
            : report
        )
      );

      alert("Report marked as in progress!");
    } catch (err: any) {
      console.error(err);

      alert(
        err?.message ||
          "Failed to start report."
      );
    }
  }

  async function handleComplete(reportId: string) {
    try {
      await updateWorkerReportStatus(
        reportId,
        "COMPLETED"
      );

      setReports((currentReports) =>
        currentReports.filter(
          (report) => report.id !== reportId
        )
      );

      alert(
        "Report marked as completed!"
      );
    } catch (err: any) {
      console.error(err);

      alert(
        err?.message ||
          "Failed to complete report."
      );
    }
  }

  if (loading) {
    return (
      <div>
        <h1>Active Reports</h1>
        <p>Loading active reports...</p>
      </div>
    );
  }

  return (
    <div>
      <h1>Active Reports</h1>

      <p style={styles.subtitle}>
        Reports currently assigned to you.
      </p>

      {error && (
        <div style={styles.error}>
          {error}

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
            No active reports.
          </div>
        ) : (
          reports.map((report) => (
            <div
              key={report.id}
              style={styles.card}
            >
              <div style={styles.header}>
                <h2>Waste Report</h2>

                <span
                  style={{
                    ...styles.status,
                    background:
                      report.status ===
                      "ACCEPTED"
                        ? "#fef3c7"
                        : "#dbeafe",

                    color:
                      report.status ===
                      "ACCEPTED"
                        ? "#92400e"
                        : "#1e40af",
                  }}
                >
                  {report.status}
                </span>
              </div>

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
                <strong>Updated:</strong>{" "}
                {new Date(
                  report.updated_at
                ).toLocaleString()}
              </p>

              {report.status ===
                "ACCEPTED" && (
                <button
                  onClick={() =>
                    handleStart(report.id)
                  }
                  style={styles.button}
                >
                  Start Work
                </button>
              )}

              {report.status ===
                "IN_PROGRESS" && (
                <button
                  onClick={() =>
                    handleComplete(report.id)
                  }
                  style={styles.button}
                >
                  Mark Completed
                </button>
              )}
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
    maxWidth: "800px",
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
  },

  status: {
    padding: "6px 12px",
    borderRadius: "20px",
    fontWeight: "bold",
    fontSize: "13px",
  },

  button: {
    marginTop: "10px",
    padding: "10px 18px",
    border: "none",
    borderRadius: "7px",
    background: "#15803d",
    color: "white",
    fontSize: "15px",
    cursor: "pointer",
  },

  empty: {
    padding: "30px",
    background: "white",
    borderRadius: "10px",
    textAlign: "center" as const,
    color: "#6b7280",
  },

  error: {
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

export default ActiveReports;