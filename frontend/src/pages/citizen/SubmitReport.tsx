import { useState } from "react";

function SubmitReport() {
  const [image, setImage] = useState<File | null>(null);
  const [description, setDescription] = useState("");
  const [address, setAddress] = useState("");
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Mock submission for now
    setSubmitted(true);
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

        <label>Address</label>

        <input
          type="text"
          placeholder="Enter waste location"
          value={address}
          onChange={(e) =>
            setAddress(e.target.value)
          }
          required
        />

        <label>Latitude</label>

        <input
          type="number"
          step="any"
          placeholder="e.g. 12.9716"
          value={latitude}
          onChange={(e) =>
            setLatitude(e.target.value)
          }
          required
        />

        <label>Longitude</label>

        <input
          type="number"
          step="any"
          placeholder="e.g. 77.5946"
          value={longitude}
          onChange={(e) =>
            setLongitude(e.target.value)
          }
          required
        />

        <button
          type="submit"
          style={styles.button}
        >
          Submit Report
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
  },
};

export default SubmitReport;