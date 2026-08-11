import { NavLink, useNavigate } from "react-router-dom";

function Sidebar() {
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/login");
  };

  return (
    <aside style={styles.sidebar}>
      <div>
        <h2 style={styles.logo}>SPIRO</h2>

        <p style={styles.subtitle}>
          Waste Management
        </p>
      </div>

      <nav style={styles.nav}>
        <NavLink
          to="/citizen/dashboard"
          style={linkStyle}
        >
          🏠 Dashboard
        </NavLink>

        <NavLink
          to="/citizen/report"
          style={linkStyle}
        >
          📝 Submit Report
        </NavLink>

        <NavLink
          to="/citizen/reports"
          style={linkStyle}
        >
          📋 My Reports
        </NavLink>
      </nav>

      <button
        onClick={handleLogout}
        style={styles.logout}
      >
        🚪 Logout
      </button>
    </aside>
  );
}

const linkStyle = ({ isActive }: { isActive: boolean }) => ({
  display: "block",
  padding: "12px 15px",
  marginBottom: "8px",
  borderRadius: "8px",
  textDecoration: "none",
  color: isActive ? "white" : "#374151",
  background: isActive ? "#15803d" : "transparent",
});

const styles = {
  sidebar: {
    width: "240px",
    minHeight: "100vh",
    padding: "25px 15px",
    background: "#ffffff",
    borderRight: "1px solid #e5e7eb",
    display: "flex",
    flexDirection: "column" as const,
  },

  logo: {
    color: "#15803d",
    margin: 0,
  },

  subtitle: {
    color: "#6b7280",
    fontSize: "13px",
    marginTop: "5px",
  },

  nav: {
    marginTop: "40px",
    flex: 1,
  },

  logout: {
    padding: "12px",
    border: "none",
    borderRadius: "8px",
    background: "#dc2626",
    color: "white",
    fontSize: "15px",
  },
};

export default Sidebar;