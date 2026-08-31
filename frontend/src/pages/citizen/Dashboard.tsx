import { useEffect, useState } from "react";
import { getMyReports, type Report } from "../../api/reports";

function Dashboard() {
  const [total, setTotal] = useState(0);
  const [pending, setPending] = useState(0);
  const [completed, setCompleted] = useState(0);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    loadStats();
  }, []);

  async function loadStats() {
    try {
      setLoading(true);
      setError("");

      const reports: Report[] = await getMyReports();

      setTotal(reports.length);

      setPending(
        reports.filter(
          (report) => report.status === "PENDING"
        ).length
      );

      setCompleted(
        reports.filter(
          (report) => report.status === "COMPLETED"
        ).length
      );
    } catch (err: any) {
      console.error(
        "Failed to load report statistics:",
        err
      );

      setError(
        err?.message ||
          "Failed to load report statistics."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={styles.container}>
      <h1>Citizen Dashboard</h1>

      <p style={styles.subtitle}>
        Welcome to SPIRO!
      </p>

      {loading && (
        <p>Loading report statistics...</p>
      )}

      {error && (
        <div style={styles.error}>
          {error}

          <button
            onClick={loadStats}
            style={styles.retryButton}
          >
            Retry
          </button>
        </div>
      )}

      {!loading && !error && (
        <div style={styles.cards}>
          <div style={styles.card}>
            <h3>Total Reports</h3>
            <h2>{total}</h2>
          </div>

          <div style={styles.card}>
            <h3>Pending</h3>
            <h2>{pending}</h2>
          </div>

          <div style={styles.card}>
            <h3>Completed</h3>
            <h2>{completed}</h2>
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    padding: "40px",
  },

  subtitle: {
    color: "#6b7280",
    marginBottom: "30px",
  },

  cards: {
    display: "flex",
    gap: "20px",
    marginTop: "30px",
    flexWrap: "wrap" as const,
  },

  card: {
    padding: "25px",
    background: "white",
    borderRadius: "12px",
    boxShadow:
      "0 4px 15px rgba(0,0,0,0.08)",
    minWidth: "180px",
  },

  error: {
    padding: "15px",
    background: "#fee2e2",
    color: "#b91c1c",
    borderRadius: "8px",
    maxWidth: "600px",
  },

  retryButton: {
    marginLeft: "15px",
    padding: "8px 14px",
    border: "none",
    borderRadius: "6px",
    background: "#15803d",
    color: "white",
    cursor: "pointer",
  },
};

export default Dashboard;