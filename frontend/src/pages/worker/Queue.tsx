import { useState } from "react";

const initialReports = [
  {
    reportId: "report-001",
    status: "PENDING",
    address: "12A MG Road",
    prediction: "Plastic",
    createdAt: "27 July 2026, 10:30 AM",
  },
  {
    reportId: "report-004",
    status: "PENDING",
    address: "Church Street",
    prediction: "Paper",
    createdAt: "27 July 2026, 11:15 AM",
  },
  {
    reportId: "report-005",
    status: "PENDING",
    address: "Indiranagar",
    prediction: "Organic",
    createdAt: "27 July 2026, 12:00 PM",
  },
];

function Queue() {
  const [reports, setReports] =
    useState(initialReports);

  const handleAccept = (reportId: string) => {
    setReports((currentReports) =>
      currentReports.filter(
        (report) =>
          report.reportId !== reportId
      )
    );

    alert("Report accepted successfully!");
  };

  return (
    <div>
      <h1>Report Queue</h1>

      <p style={styles.subtitle}>
        Pending reports from your assigned wards.
      </p>

      <div style={styles.container}>
        {reports.length === 0 ? (
          <div style={styles.empty}>
            No pending reports.
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
                <strong>Reported:</strong>{" "}
                {report.createdAt}
              </p>

              <button
                onClick={() =>
                  handleAccept(report.reportId)
                }
                style={styles.button}
              >
                Accept Report
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
    background: "#fef3c7",
    color: "#92400e",
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

export default Queue;