import { useState } from "react";

const initialReports = [
  {
    reportId: "report-010",
    status: "IN_PROGRESS",
    address: "MG Road",
    prediction: "Plastic",
    acceptedAt: "27 July 2026, 11:00 AM",
  },
  {
    reportId: "report-011",
    status: "IN_PROGRESS",
    address: "Koramangala",
    prediction: "Organic",
    acceptedAt: "27 July 2026, 11:30 AM",
  },
];

function ActiveReports() {
  const [reports, setReports] =
    useState(initialReports);

  const handleComplete = (reportId: string) => {
    setReports((currentReports) =>
      currentReports.filter(
        (report) =>
          report.reportId !== reportId
      )
    );

    alert(
      "Report marked as completed!"
    );
  };

  return (
    <div>
      <h1>Active Reports</h1>

      <p style={styles.subtitle}>
        Reports currently assigned to you.
      </p>

      <div style={styles.container}>
        {reports.length === 0 ? (
          <div style={styles.empty}>
            No active reports.
          </div>
        ) : (
          reports.map((report) => (
            <div
              key={report.reportId}
              style={styles.card}
            >
              <div style={styles.header}>
                <h2>
                  {report.prediction} Waste
                </h2>

                <span style={styles.status}>
                  {report.status}
                </span>
              </div>

              <p>
                <strong>Address:</strong>{" "}
                {report.address}
              </p>

              <p>
                <strong>Accepted:</strong>{" "}
                {report.acceptedAt}
              </p>

              <button
                onClick={() =>
                  handleComplete(
                    report.reportId
                  )
                }
                style={styles.button}
              >
                Mark Completed
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
    background: "#dbeafe",
    color: "#1e40af",
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
};

export default ActiveReports;