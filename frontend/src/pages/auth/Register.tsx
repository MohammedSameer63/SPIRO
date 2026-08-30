import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { register } from "../../api/auth";

function Register() {
  const navigate = useNavigate();

  const [wardId, setWardId] = useState("");
  const [houseNumber, setHouseNumber] = useState("");
  const [streetName, setStreetName] = useState("");
  const [address, setAddress] = useState("");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");
  const [password, setPassword] = useState("");

  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleRegister = async (
    e: React.FormEvent<HTMLFormElement>
  ) => {
    e.preventDefault();

    setError("");
    setLoading(true);

    try {
      const response = await register({
        ward_id: wardId,
        house_number: houseNumber,
        street_name: streetName,
        address,
        name,
        email,
        phone: phone || undefined,
        password,
      });

      console.log("Registration response:", response);

      alert("Registration successful! Please login.");

      navigate("/login");
    } catch (error: any) {
      console.error("Registration error:", error);

      setError(
        error?.message ||
          "Registration failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>

        <h1 style={styles.title}>SPIRO</h1>

        <p style={styles.subtitle}>
          Smart Waste Management System
        </p>

        <h2 style={styles.heading}>
          Create your account
        </h2>

        <form onSubmit={handleRegister}>

          <div style={styles.field}>
            <label>Ward ID</label>

            <input
              type="text"
              placeholder="Enter ward ID"
              value={wardId}
              onChange={(e) =>
                setWardId(e.target.value)
              }
              required
            />
          </div>

          <div style={styles.field}>
            <label>House Number</label>

            <input
              type="text"
              placeholder="Enter house number"
              value={houseNumber}
              onChange={(e) =>
                setHouseNumber(e.target.value)
              }
              required
            />
          </div>

          <div style={styles.field}>
            <label>Street Name</label>

            <input
              type="text"
              placeholder="Enter street name"
              value={streetName}
              onChange={(e) =>
                setStreetName(e.target.value)
              }
              required
            />
          </div>

          <div style={styles.field}>
            <label>Full Address</label>

            <textarea
              placeholder="Enter your full address"
              value={address}
              onChange={(e) =>
                setAddress(e.target.value)
              }
              required
            />
          </div>

          <div style={styles.field}>
            <label>Name</label>

            <input
              type="text"
              placeholder="Enter your name"
              value={name}
              onChange={(e) =>
                setName(e.target.value)
              }
              required
            />
          </div>

          <div style={styles.field}>
            <label>Email</label>

            <input
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) =>
                setEmail(e.target.value)
              }
              required
            />
          </div>

          <div style={styles.field}>
            <label>Phone</label>

            <input
              type="tel"
              placeholder="Enter your phone number"
              value={phone}
              onChange={(e) =>
                setPhone(e.target.value)
              }
            />
          </div>

          <div style={styles.field}>
            <label>Password</label>

            <input
              type="password"
              placeholder="Create a password"
              value={password}
              onChange={(e) =>
                setPassword(e.target.value)
              }
              required
            />
          </div>

          {error && (
            <p style={styles.error}>
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={loading}
            style={styles.button}
          >
            {loading
              ? "Registering..."
              : "Register"}
          </button>

        </form>

        <p style={styles.loginText}>
          Already have an account?{" "}
          <Link to="/login">
            Login
          </Link>
        </p>

      </div>
    </div>
  );
}

const styles = {
  container: {
    minHeight: "100vh",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    background: "#f4f7f5",
  },

  card: {
    width: "400px",
    padding: "40px",
    background: "white",
    borderRadius: "12px",
    boxShadow:
      "0 10px 30px rgba(0,0,0,0.1)",
  },

  title: {
    margin: "0",
    fontSize: "36px",
    color: "#15803d",
  },

  subtitle: {
    color: "#6b7280",
    marginTop: "5px",
    marginBottom: "30px",
  },

  heading: {
    marginBottom: "25px",
  },

  field: {
    display: "flex",
    flexDirection: "column" as const,
    gap: "8px",
    marginBottom: "18px",
  },

  button: {
    width: "100%",
    padding: "12px",
    background: "#15803d",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    fontSize: "16px",
  },

  error: {
    color: "#dc2626",
    marginBottom: "15px",
  },

  loginText: {
    marginTop: "20px",
    textAlign: "center" as const,
  },
};

export default Register;