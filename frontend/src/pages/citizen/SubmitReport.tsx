import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { createReport } from "../../api/reports";

function SubmitReport() {
  const navigate = useNavigate();

  const [image, setImage] = useState<File | null>(null);
  const [description, setDescription] = useState("");

  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (
    e: React.FormEvent<HTMLFormElement>
  ) => {
    e.preventDefault();

    setError("");
    setSubmitted(false);

    if (!image) {
      setError("Please select a waste image.");
      return;
    }

    if (!description.trim()) {
      setError("Please describe the waste.");
      return;
    }

    setLoading(true);

    try {
      await createReport(
        image,
        description
      );

      setSubmitted(true);

      setImage(null);
      setDescription("");

      setTimeout(() => {
        navigate("/citizen/reports");
      }, 1000);
    } catch (error: any) {
      setError(
        error?.message ||
          "Failed to submit report."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Submit Waste Report</h1>

      <p style={styles.subtitle}>
        Report waste that needs to be collected.
      </p>

      {submitted && (
        <div style={styles.success}>
          Report submitted successfully!
          <br />
          Status: PENDING
        </div>
      )}

      {error && (
        <div style={styles.error}>
          {error}
        </div>
      )}

      <form
        onSubmit={handleSubmit}
        style={styles.form}
      >
        <label>Waste Image</label>

        <input
          type="file"
          accept="image/*"
          onChange={(e) =>
            setImage(
              e.target.files?.[0] || null
            )
          }
          required
        />

        {image && (
          <p>
            Selected: {image.name}
          </p>
        )}

        <label>Description</label>

        <textarea
          placeholder="Describe the waste..."
          value={description}
          onChange={(e) =>
            setDescription(e.target.value)
          }
          required
        />

        <button
          type="submit"
          style={styles.button}
          disabled={loading}
        >
          {loading
            ? "Submitting..."
            : "Submit Report"}
        </button>
      </form>
    </div>
  );
}

const styles = {
  subtitle: {
    color: "#6b7280",
    marginBottom: "25px",
  },

  success: {
    padding: "15px",
    marginBottom: "20px",
    borderRadius: "8px",
    background: "#dcfce7",
    color: "#166534",
  },

  error: {
    padding: "15px",
    marginBottom: "20px",
    borderRadius: "8px",
    background: "#fee2e2",
    color: "#991b1b",
  },

  form: {
    maxWidth: "600px",
    display: "flex",
    flexDirection: "column" as const,
    gap: "10px",
  },

  button: {
    marginTop: "15px",
    padding: "12px",
    border: "none",
    borderRadius: "8px",
    background: "#15803d",
    color: "white",
    fontSize: "16px",
    cursor: "pointer",
  },
};

export default SubmitReport;