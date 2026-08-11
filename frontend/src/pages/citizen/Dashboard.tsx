function Dashboard() {
  return (
    <div style={{ padding: "40px" }}>
      <h1>Citizen Dashboard</h1>

      <p>Welcome to SPIRO!</p>

      <div
        style={{
          display: "flex",
          gap: "20px",
          marginTop: "30px",
        }}
      >
        <div style={cardStyle}>
          <h3>Total Reports</h3>
          <h2>18</h2>
        </div>

        <div style={cardStyle}>
          <h3>Pending</h3>
          <h2>2</h2>
        </div>

        <div style={cardStyle}>
          <h3>Completed</h3>
          <h2>16</h2>
        </div>
      </div>
    </div>
  );
}

const cardStyle = {
  padding: "25px",
  background: "white",
  borderRadius: "12px",
  boxShadow: "0 4px 15px rgba(0,0,0,0.08)",
  minWidth: "180px",
};

export default Dashboard;