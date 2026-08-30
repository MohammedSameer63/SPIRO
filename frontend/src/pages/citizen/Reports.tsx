import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getMyReports } from "../../api/reports";
import type { Report } from "../../api/reports";

function Reports() {
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

      const response = await getMyReports();

      setReports(response);
      
    } catch (err: any) {
      console.error("Failed to load reports:", err);

      setError(
        err?.message || "Failed to load reports."
      );
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div>
        <h1>My Reports</h1>

        <p style={styles.subtitle}>
          View all the waste reports you have submitted.
        </p>

        <p>Loading reports...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <h1>My Reports</h1>

        <p style={styles.subtitle}>
          View all the waste reports you have submitted.
        </p>

        <div style={styles.error}>
          {error}
        </div>

        <button
          onClick={loadReports}
          style={styles.retryButton}
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div>
      <h1>My Reports</h1>

      <p style={styles.subtitle}>
        View all the waste reports you have submitted.
      </p>

      {reports.length === 0 ? (
        <div style={styles.empty}>
          <h2>No reports yet</h2>

          <p>
            You have not submitted any waste reports.
          </p>

          <Link
            to="/citizen/report"
            style={styles.button}
          >
            Submit Your First Report
          </Link>
        </div>
      ) : (
        <div style={styles.container}>
          {reports.map((report) => (
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
                      report.status === "COMPLETED"
                        ? "#dcfce7"
                        : report.status === "PENDING"
                        ? "#fef3c7"
                        : "#dbeafe",

                    color:
                      report.status === "COMPLETED"
                        ? "#166534"
                        : report.status === "PENDING"
                        ? "#92400e"
                        : "#1e40af",
                  }}
                >
                  {report.status}
                </span>
              </div>

              <p>
                <strong>Description:</strong>{" "}
                {report.description || "No description"}
              </p>

              <p>
                <strong>Address:</strong>{" "}
                {report.address || "No address"}
              </p>

              <p>
                <strong>Location:</strong>{" "}
                {report.latitude}, {report.longitude}
              </p>

              <p>
                <strong>Reported:</strong>{" "}
                {new Date(
                  report.created_at
                ).toLocaleString()}
              </p>

              <Link
                to={`/citizen/reports/${report.id}`}
                style={styles.button}
              >
                View Details
              </Link>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const styles = {
  subtitle: {
    color: "#6b7280",
    marginBottom: "25px",
  },

  container: {
    display: "flex",
    flexDirection: "column" as const,
    gap: "20px",
    maxWidth: "800px",
  },

  card: {
    padding: "25px",
    background: "white",
    borderRadius: "12px",
    boxShadow: "0 4px 15px rgba(0,0,0,0.08)",
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
    fontSize: "13px",
    fontWeight: "bold",
  },

  button: {
    display: "inline-block",
    marginTop: "10px",
    padding: "10px 15px",
    background: "#15803d",
    color: "white",
    borderRadius: "7px",
    textDecoration: "none",
  },

  empty: {
    padding: "40px",
    background: "white",
    borderRadius: "12px",
    textAlign: "center" as const,
    boxShadow: "0 4px 15px rgba(0,0,0,0.08)",
  },

  error: {
    padding: "15px",
    marginBottom: "15px",
    borderRadius: "8px",
    background: "#fee2e2",
    color: "#b91c1c",
  },

  retryButton: {
    padding: "10px 18px",
    border: "none",
    borderRadius: "7px",
    background: "#15803d",
    color: "white",
    cursor: "pointer",
  },
};

export default Reports;