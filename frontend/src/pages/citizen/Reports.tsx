import { Link } from "react-router-dom";

const reports = [
  {
    id: "report-001",
    category: "Plastic",
    address: "MG Road, Bangalore",
    status: "PENDING",
    confidence: "94.20%",
    date: "27 July 2026",
  },
  {
    id: "report-002",
    category: "Paper",
    address: "Church Street, Bangalore",
    status: "COMPLETED",
    confidence: "91.50%",
    date: "26 July 2026",
  },
  {
    id: "report-003",
    category: "Organic",
    address: "Indiranagar, Bangalore",
    status: "IN_PROGRESS",
    confidence: "96.10%",
    date: "25 July 2026",
  },
];

function Reports() {
  return (
    <div>
      <h1>My Reports</h1>

      <p style={styles.subtitle}>
        View all the waste reports you have submitted.
      </p>

      <div style={styles.container}>
        {reports.map((report) => (
          <div key={report.id} style={styles.card}>
            <div style={styles.header}>
              <h2>{report.category} Waste</h2>

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
              <strong>Address:</strong> {report.address}
            </p>

            <p>
              <strong>AI Confidence:</strong>{" "}
              {report.confidence}
            </p>

            <p>
              <strong>Reported:</strong> {report.date}
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
};

export default Reports;