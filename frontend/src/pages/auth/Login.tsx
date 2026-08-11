import { login } from "../../api/auth";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

function Login() {
  const navigate = useNavigate();

  const [email, setEmail] =
              useState("");

  const [password, setPassword] =
             useState("");

  const [error, setError] =
             useState("");

  const [loading, setLoading] =
             useState(false);
  
   const handleLogin = async (
  e: React.FormEvent
) => {
  e.preventDefault();

  setError("");
  setLoading(true);

  try {
    const response = await login({
      email,
      password,
    });

    const { token, user } =
      response.data;

    localStorage.setItem(
      "token",
      token
    );

    localStorage.setItem(
      "user",
      JSON.stringify(user)
    );

    localStorage.setItem(
      "role",
      user.role
    );

    if (user.role === "CITIZEN") {
      navigate("/citizen/dashboard");
    } else if (
      user.role === "WORKER"
    ) {
      navigate("/worker/dashboard");
    } else if (
      user.role === "ADMIN"
    ) {
      navigate("/admin/dashboard");
    }
  } catch (error: any) {
    setError(
      error.message ||
        "Login failed."
    );
  } finally {
    setLoading(false);
  }
};
  const navigate = useNavigate();

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();

    // Mock login for now
    localStorage.setItem("token", "mock-jwt-token");

    navigate("/citizen/dashboard");
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h1>SPIRO</h1>

        <p style={styles.subtitle}>Smart Waste Management System</p>

        <h2>Welcome Back</h2>

        <form onSubmit={handleLogin}>
          <label>Email</label>

          <input
  type="email"
  value={email}
  onChange={(e) =>
    setEmail(e.target.value)
  }
  placeholder="Enter your email"
  required
/>

          <label>Password</label>

          <input
  type="password"
  value={password}
  onChange={(e) =>
    setPassword(e.target.value)
  }
  placeholder="Enter your password"
  required
/>

<button
  type="submit"
  disabled={loading}
>
  {loading
    ? "Logging in..."
    : "Login"}
</button>

          <button type="submit">Login</button>

          {error && (
  <p style={{ color: "red" }}>
    {error}
  </p>
)}

        </form>

        <p style={styles.registerText}>
          Don't have an account?{" "}
          <Link to="/register">Register</Link>
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
    boxShadow: "0 10px 30px rgba(0,0,0,0.1)",
  },

  subtitle: {
    color: "#6b7280",
    marginBottom: "30px",
  },

  registerText: {
    marginTop: "20px",
    textAlign: "center" as const,
  },
};

export default Login;