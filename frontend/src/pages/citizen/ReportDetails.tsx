import { Link, useParams } from "react-router-dom";

function ReportDetails() {
  const { id } = useParams();

  return (
    <div>
      <Link
        to="/citizen/reports"
        style={styles.back}
      >
        ← Back to My Reports
      </Link>

      <h1 style={styles.title}>Report Details</h1>

      <div style={styles.card}>
        <div style={styles.imagePlaceholder}>
          Waste Image
        </div>

        <h2>Plastic Waste</h2>

        <div style={styles.status}>
          PENDING
        </div>

        <div style={styles.section}>
          <h3>AI Prediction</h3>

          <p>
            <strong>Category:</strong> Plastic
          </p>

          <p>
            <strong>Confidence:</strong> 94.20%
          </p>
        </div>

        <div style={styles.section}>
          <h3>Location</h3>

          <p>
            <strong>Address:</strong> MG Road, Bangalore
          </p>

          <p>
            <strong>Latitude:</strong> 12.9716
          </p>

          <p>
            <strong>Longitude:</strong> 77.5946
          </p>
        </div>

        <div style={styles.section}>
          <h3>Description</h3>

          <p>
            Plastic waste found near the roadside.
          </p>
        </div>

        <div style={styles.section}>
          <h3>Report Information</h3>

          <p>
            <strong>Report ID:</strong> {id}
          </p>

          <p>
            <strong>Reported:</strong> 27 July 2026
          </p>
        </div>
      </div>
    </div>
  );
}

const styles = {
  back: {
    color: "#15803d",
    textDecoration: "none",
  },

  title: {
    marginTop: "20px",
  },

  card: {
    maxWidth: "800px",
    marginTop: "25px",
    padding: "30px",
    background: "white",
    borderRadius: "12px",
    boxShadow: "0 4px 15px rgba(0,0,0,0.08)",
  },

  imagePlaceholder: {
    height: "250px",
    background: "#e5e7eb",
    borderRadius: "10px",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: "25px",
    color: "#6b7280",
    fontSize: "18px",
  },

  status: {
    display: "inline-block",
    padding: "7px 14px",
    borderRadius: "20px",
    background: "#fef3c7",
    color: "#92400e",
    fontWeight: "bold",
    marginTop: "10px",
  },

  section: {
    marginTop: "25px",
    paddingTop: "15px",
    borderTop: "1px solid #e5e7eb",
  },
};

export default ReportDetails;