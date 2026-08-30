import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  getReport,
  type Report,
} from "../../api/reports";

const API_SERVER_URL = "http://127.0.0.1:8000";

function ReportDetails() {
  const { id } = useParams();

  const [report, setReport] =
    useState<Report | null>(null);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  useEffect(() => {
    if (id) {
      loadReport(id);
    }
  }, [id]);

  async function loadReport(reportId: string) {
    try {
      setLoading(true);
      setError("");

      const response = await getReport(reportId);

      setReport(response);
      
    } catch (error: any) {
      setError(
        error?.message ||
          "Failed to load report."
      );
    } finally {
      setLoading(false);
    }
  }

  function formatDate(date: string) {
    return new Date(date).toLocaleString(
      "en-IN",
      {
        day: "numeric",
        month: "long",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      }
    );
  }

  function getImageUrl(imageUrl: string) {
    if (!imageUrl) {
      return "";
    }

    if (imageUrl.startsWith("http://") || imageUrl.startsWith("https://")) {
      return imageUrl;
    }
    return `${API_SERVER_URL}${imageUrl.startsWith("/") ? "" : "/"}${imageUrl}`;
  }

  function getStatusStyle(
    status: Report["status"]
  ) {
    if (status === "COMPLETED") {
      return {
        background: "#dcfce7",
        color: "#166534",
      };
    }

    if (status === "PENDING") {
      return {
        background: "#fef3c7",
        color: "#92400e",
      };
    }

    if (status === "IN_PROGRESS") {
      return {
        background: "#dbeafe",
        color: "#1e40af",
      };
    }

    return {
      background: "#f3e8ff",
      color: "#7e22ce",
    };
  }

  if (loading) {
    return (
      <div>
        <p>Loading report...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <Link
          to="/citizen/reports"
          style={styles.back}
        >
          ← Back to My Reports
        </Link>

        <div style={styles.error}>
          {error}
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div>
        <p>Report not found.</p>

        <Link
          to="/citizen/reports"
          style={styles.back}
        >
          ← Back to My Reports
        </Link>
      </div>
    );
  }

  const imageUrl = report.image_url
    ? getImageUrl(report.image_url)
    : "";

  return (
    <div>
      <Link
        to="/citizen/reports"
        style={styles.back}
      >
        ← Back to My Reports
      </Link>

      <h1>Report Details</h1>

      <p style={styles.subtitle}>
        Detailed information about your
        waste report.
      </p>

      <div style={styles.card}>
        <div style={styles.header}>
          <h2>Waste Report</h2>

          <span
            style={{
              ...styles.status,
              ...getStatusStyle(
                report.status
              ),
            }}
          >
            {report.status}
          </span>
        </div>

        {/* WASTE IMAGE */}
        {imageUrl && (
          <div style={styles.imageSection}>
            <h3>Waste Image</h3>

            <img
              src={imageUrl}
              alt="Reported waste"
              style={styles.image}
              onError={(e) => {
                e.currentTarget.style.display =
                  "none";
              }}
            />
          </div>
        )}

        <div style={styles.section}>
          <h3>Waste Information</h3>

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
        </div>

        <div style={styles.section}>
          <h3>Location</h3>

          <p>
            <strong>Latitude:</strong>{" "}
            {report.latitude}
          </p>

          <p>
            <strong>Longitude:</strong>{" "}
            {report.longitude}
          </p>
        </div>

        <div style={styles.section}>
          <h3>Report Status</h3>

          <p>
            <strong>Status:</strong>{" "}
            {report.status}
          </p>

          <p>
            <strong>Submitted:</strong>{" "}
            {formatDate(
              report.created_at
            )}
          </p>

          <p>
            <strong>Last Updated:</strong>{" "}
            {formatDate(
              report.updated_at
            )}
          </p>
        </div>
      </div>
    </div>
  );
}

const styles = {
  back: {
    display: "inline-block",
    marginBottom: "20px",
    color: "#15803d",
    textDecoration: "none",
  },

  subtitle: {
    color: "#6b7280",
    marginBottom: "25px",
  },

  card: {
    maxWidth: "800px",
    padding: "30px",
    background: "white",
    borderRadius: "12px",
    boxShadow:
      "0 4px 15px rgba(0,0,0,0.08)",
  },

  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "25px",
  },

  status: {
    padding: "7px 14px",
    borderRadius: "20px",
    fontSize: "13px",
    fontWeight: "bold",
  },

  imageSection: {
    marginBottom: "30px",
  },

  image: {
    width: "100%",
    maxWidth: "700px",
    maxHeight: "500px",
    objectFit: "contain" as const,
    borderRadius: "10px",
    border: "1px solid #e5e7eb",
  },

  section: {
    padding: "20px 0",
    borderTop:
      "1px solid #e5e7eb",
  },

  error: {
    padding: "15px",
    marginTop: "20px",
    borderRadius: "8px",
    background: "#fee2e2",
    color: "#991b1b",
  },
};

export default ReportDetails;